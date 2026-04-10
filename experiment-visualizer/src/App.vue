<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

type EventRecord = {
  event_type?: string
  timestamp?: string
  iteration_index?: number
  execution_status?: string
  action_proposed?: {
    metadata?: {
      command?: string
    }
  }
  action_executed?: {
    metadata?: {
      command?: string
    }
  }
  [key: string]: unknown
}

type SummaryRecord = {
  run_id?: string
  started_at?: string
  ended_at?: string
  duration_ms?: number
  success?: boolean
  termination_reason?: string
  iterations_total?: number
  llm_provider?: string
  llm_model?: string
  goal_text?: string
  [key: string]: unknown
}

type ExperimentRun = {
  key: string
  folderName: string
  summary?: SummaryRecord
  events: EventRecord[]
  parseWarnings: string[]
}

const runs = ref<ExperimentRun[]>([])
const selectedRunKey = ref<string>('')
const isParsing = ref(false)
const parseMessage = ref<string>('')

const selectedRun = computed<ExperimentRun | undefined>(() => {
  return runs.value.find((run) => run.key === selectedRunKey.value) ?? runs.value[0]
})

const selectedEventCounts = computed<Array<{ name: string; count: number }>>(() => {
  const currentRun = selectedRun.value
  if (!currentRun) {
    return []
  }

  const counts = new Map<string, number>()
  for (const event of currentRun.events) {
    const eventType = event.event_type ?? 'unknown'
    counts.set(eventType, (counts.get(eventType) ?? 0) + 1)
  }

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
})

const selectedActionCounts = computed<Array<{ name: string; count: number }>>(() => {
  const currentRun = selectedRun.value
  if (!currentRun) {
    return []
  }

  const counts = new Map<string, number>()
  for (const event of currentRun.events) {
    const command =
      event.action_proposed?.metadata?.command ?? event.action_executed?.metadata?.command
    if (!command) {
      continue
    }
    counts.set(command, (counts.get(command) ?? 0) + 1)
  }

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
})

const maxEventCount = computed<number>(() => {
  return Math.max(1, ...selectedEventCounts.value.map((entry) => entry.count))
})

const maxActionCount = computed<number>(() => {
  return Math.max(1, ...selectedActionCounts.value.map((entry) => entry.count))
})

const runDurationMs = computed<number>(() => {
  const summary = selectedRun.value?.summary
  if (!summary) {
    return 0
  }

  if (typeof summary.duration_ms === 'number') {
    return summary.duration_ms
  }

  return Math.max(0, asEpochMs(summary.ended_at) - asEpochMs(summary.started_at))
})

const timelineRows = computed<EventRecord[]>(() => {
  return selectedRun.value?.events ?? []
})

function asEpochMs(value: string | undefined): number {
  if (!value) {
    return 0
  }
  const epoch = Date.parse(value)
  return Number.isFinite(epoch) ? epoch : 0
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : undefined
  }
  return undefined
}

function asSummaryRecord(value: unknown): SummaryRecord | undefined {
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    return undefined
  }
  return value as SummaryRecord
}

function asEventRecord(value: unknown): EventRecord | undefined {
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    return undefined
  }
  return value as EventRecord
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown
  } catch {
    return undefined
  }
}

function formatMs(durationMs: number): string {
  if (!durationMs || durationMs < 0) {
    return '0 ms'
  }
  if (durationMs < 1000) {
    return `${durationMs.toFixed(0)} ms`
  }
  if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(2)} s`
  }
  const minutes = Math.floor(durationMs / 60000)
  const seconds = ((durationMs % 60000) / 1000).toFixed(1)
  return `${minutes}m ${seconds}s`
}

function formatDate(timestamp: string | undefined): string {
  if (!timestamp) {
    return 'n/a'
  }

  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) {
    return timestamp
  }

  return date.toLocaleString()
}

function formatStatus(success: boolean | undefined): string {
  if (success === true) {
    return 'Success'
  }
  if (success === false) {
    return 'Failed'
  }
  return 'Unknown'
}

function normalizePath(file: File): string {
  const webkitRelativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath
  if (webkitRelativePath && webkitRelativePath.length > 0) {
    return webkitRelativePath
  }
  return file.name
}

async function parseExperimentFiles(files: FileList): Promise<ExperimentRun[]> {
  const grouped = new Map<string, ExperimentRun>()

  for (const file of Array.from(files)) {
    const relativePath = normalizePath(file)
    const pieces = relativePath.split('/').filter(Boolean)
    const fileName = pieces[pieces.length - 1]

    if (fileName !== 'summary.json' && fileName !== 'events.jsonl') {
      continue
    }

    const parentPath = pieces.slice(0, -1).join('/')
    const folderName = pieces.length > 1 ? pieces[pieces.length - 2] : 'unknown-run'
    const runKey = parentPath || folderName

    if (!grouped.has(runKey)) {
      grouped.set(runKey, {
        key: runKey,
        folderName,
        events: [],
        parseWarnings: [],
      })
    }

    const run = grouped.get(runKey)
    if (!run) {
      continue
    }

    const content = await file.text()

    if (fileName === 'summary.json') {
      const parsed = parseJson(content)
      const summary = asSummaryRecord(parsed)
      if (summary) {
        run.summary = summary
      } else {
        run.parseWarnings.push(`Invalid JSON in ${relativePath}`)
      }
      continue
    }

    const lines = content.split('\n')
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index].trim()
      if (!line) {
        continue
      }

      const parsed = parseJson(line)
      const event = asEventRecord(parsed)
      if (!event) {
        run.parseWarnings.push(`Invalid line ${index + 1} in ${relativePath}`)
        continue
      }

      run.events.push(event)
    }
  }

  return [...grouped.values()].sort((a, b) => {
    const aStart = asEpochMs(a.summary?.started_at)
    const bStart = asEpochMs(b.summary?.started_at)

    if (aStart === bStart) {
      return a.folderName.localeCompare(b.folderName)
    }

    return bStart - aStart
  })
}

async function handleDirectoryPick(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const files = input.files

  if (!files || files.length === 0) {
    return
  }

  isParsing.value = true
  parseMessage.value = ''

  try {
    const parsedRuns = await parseExperimentFiles(files)
    runs.value = parsedRuns
    selectedRunKey.value = parsedRuns[0]?.key ?? ''

    if (parsedRuns.length === 0) {
      parseMessage.value =
        'No experiment files found. Pick a folder that contains run directories with summary.json and events.jsonl.'
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown parse error'
    parseMessage.value = `Failed to parse files: ${message}`
  } finally {
    isParsing.value = false
  }
}

async function loadDefaultExperiments(): Promise<void> {
  isParsing.value = true

  try {
    const response = await fetch('/api/experiments/default')
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const payload = (await response.json()) as { runs?: ExperimentRun[]; error?: string }
    const parsedRuns = Array.isArray(payload.runs) ? payload.runs : []

    if (parsedRuns.length === 0) {
      parseMessage.value =
        'No runs found in default experiments folder. You can still choose a folder manually.'
      return
    }

    runs.value = parsedRuns
    selectedRunKey.value = parsedRuns[0]?.key ?? ''
    parseMessage.value = `Loaded ${parsedRuns.length} run(s) from default experiments folder.`
  } catch {
    parseMessage.value =
      'Could not auto-load default experiments folder. Choose a folder manually to continue.'
  } finally {
    isParsing.value = false
  }
}

onMounted(async () => {
  await loadDefaultExperiments()
})

function eventValue(event: EventRecord, key: string): string {
  const raw = event[key]
  if (raw === undefined || raw === null) {
    return 'n/a'
  }

  if (typeof raw === 'object') {
    try {
      return JSON.stringify(raw)
    } catch {
      return '[object]'
    }
  }

  return String(raw)
}

function eventBarWidth(entryCount: number): string {
  return `${(entryCount / maxEventCount.value) * 100}%`
}

function actionBarWidth(entryCount: number): string {
  return `${(entryCount / maxActionCount.value) * 100}%`
}
</script>

<template>
  <main class="layout">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">Hexapod Experiment Viewer</p>
        <h1>Visualize run outcomes and event flows</h1>
        <p class="subtitle">
          Choose the experiments folder and inspect summaries, action patterns, and full event timelines.
        </p>
      </div>
      <label class="picker">
        <input
          class="picker-input"
          type="file"
          webkitdirectory
          directory
          multiple
          @change="handleDirectoryPick"
        />
        <span>{{ isParsing ? 'Parsing files...' : 'Choose Experiments Folder' }}</span>
      </label>
      <button class="picker" type="button" @click="loadDefaultExperiments">
        <span>{{ isParsing ? 'Loading default...' : 'Reload Default (experiments)' }}</span>
      </button>
    </header>

    <p v-if="parseMessage" class="message">{{ parseMessage }}</p>

    <section v-if="runs.length > 0" class="grid">
      <aside class="panel run-list">
        <h2>Runs ({{ runs.length }})</h2>
        <ul>
          <li v-for="run in runs" :key="run.key">
            <button
              type="button"
              :class="['run-chip', { active: run.key === selectedRun?.key }]"
              @click="selectedRunKey = run.key"
            >
              <strong>{{ run.summary?.run_id ?? run.folderName }}</strong>
              <span>{{ formatDate(run.summary?.started_at) }}</span>
              <span>{{ formatStatus(run.summary?.success) }}</span>
            </button>
          </li>
        </ul>
      </aside>

      <section class="panel details" v-if="selectedRun">
        <h2>Run Details</h2>

        <div class="stats">
          <article class="stat-card">
            <h3>Status</h3>
            <p>{{ formatStatus(selectedRun.summary?.success as boolean | undefined) }}</p>
          </article>
          <article class="stat-card">
            <h3>Duration</h3>
            <p>{{ formatMs(runDurationMs) }}</p>
          </article>
          <article class="stat-card">
            <h3>Iterations</h3>
            <p>{{ asNumber(selectedRun.summary?.iterations_total) ?? selectedRun.events.length }}</p>
          </article>
          <article class="stat-card">
            <h3>Model</h3>
            <p>{{ selectedRun.summary?.llm_model ?? 'n/a' }}</p>
          </article>
        </div>

        <div class="meta">
          <p><strong>Goal:</strong> {{ selectedRun.summary?.goal_text ?? 'n/a' }}</p>
          <p><strong>Provider:</strong> {{ selectedRun.summary?.llm_provider ?? 'n/a' }}</p>
          <p><strong>Termination:</strong> {{ selectedRun.summary?.termination_reason ?? 'n/a' }}</p>
          <p><strong>Started:</strong> {{ formatDate(selectedRun.summary?.started_at) }}</p>
          <p><strong>Ended:</strong> {{ formatDate(selectedRun.summary?.ended_at) }}</p>
        </div>

        <div class="bars-grid">
          <section>
            <h3>Event Distribution</h3>
            <ul class="bars">
              <li v-for="entry in selectedEventCounts" :key="entry.name">
                <span>{{ entry.name }}</span>
                <div class="bar-track">
                  <div class="bar-fill event" :style="{ width: eventBarWidth(entry.count) }"></div>
                </div>
                <strong>{{ entry.count }}</strong>
              </li>
            </ul>
          </section>

          <section>
            <h3>Action Commands</h3>
            <ul class="bars">
              <li v-for="entry in selectedActionCounts" :key="entry.name">
                <span>{{ entry.name }}</span>
                <div class="bar-track">
                  <div class="bar-fill action" :style="{ width: actionBarWidth(entry.count) }"></div>
                </div>
                <strong>{{ entry.count }}</strong>
              </li>
            </ul>
          </section>
        </div>

        <section class="timeline">
          <h3>Event Timeline ({{ timelineRows.length }})</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Type</th>
                  <th>Iteration</th>
                  <th>Command</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(event, index) in timelineRows" :key="`${eventValue(event, 'timestamp')}-${index}`">
                  <td>{{ formatDate(event.timestamp) }}</td>
                  <td>{{ event.event_type ?? 'n/a' }}</td>
                  <td>{{ event.iteration_index ?? 'n/a' }}</td>
                  <td>
                    {{
                      event.action_proposed?.metadata?.command ??
                      event.action_executed?.metadata?.command ??
                      'n/a'
                    }}
                  </td>
                  <td>{{ event.execution_status ?? 'n/a' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <p v-if="selectedRun.parseWarnings.length > 0" class="warning">
          Parse warnings: {{ selectedRun.parseWarnings.join(' | ') }}
        </p>
      </section>
    </section>

    <section v-else class="empty panel">
      <h2>No Runs Loaded</h2>
      <p>
        Pick your <strong>experiments</strong> directory. The app will look for files named
        <strong>summary.json</strong> and <strong>events.jsonl</strong> in each run folder.
      </p>
    </section>
  </main>
</template>
