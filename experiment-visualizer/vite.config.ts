import fs from 'node:fs/promises'
import path from 'node:path'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

type ParsedRun = {
  key: string
  folderName: string
  summary?: Record<string, unknown>
  events: Array<Record<string, unknown>>
  parseWarnings: string[]
}

const experimentsDir = path.resolve(__dirname, '../experiments')

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

async function loadDefaultRuns(): Promise<ParsedRun[]> {
  const output: ParsedRun[] = []
  const entries = await fs.readdir(experimentsDir, { withFileTypes: true })

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }

    const runDir = path.join(experimentsDir, entry.name)
    const summaryPath = path.join(runDir, 'summary.json')
    const eventsPath = path.join(runDir, 'events.jsonl')

    const run: ParsedRun = {
      key: entry.name,
      folderName: entry.name,
      events: [],
      parseWarnings: [],
    }

    if (await fileExists(summaryPath)) {
      try {
        const summaryText = await fs.readFile(summaryPath, 'utf-8')
        run.summary = JSON.parse(summaryText) as Record<string, unknown>
      } catch {
        run.parseWarnings.push(`Invalid JSON in ${entry.name}/summary.json`)
      }
    }

    if (await fileExists(eventsPath)) {
      try {
        const eventsText = await fs.readFile(eventsPath, 'utf-8')
        const lines = eventsText.split('\n')
        for (let index = 0; index < lines.length; index += 1) {
          const line = lines[index].trim()
          if (!line) {
            continue
          }
          try {
            run.events.push(JSON.parse(line) as Record<string, unknown>)
          } catch {
            run.parseWarnings.push(`Invalid line ${index + 1} in ${entry.name}/events.jsonl`)
          }
        }
      } catch {
        run.parseWarnings.push(`Failed to read ${entry.name}/events.jsonl`)
      }
    }

    if (run.summary || run.events.length > 0) {
      output.push(run)
    }
  }

  output.sort((a, b) => {
    const aStart = Date.parse(String(a.summary?.started_at ?? '')) || 0
    const bStart = Date.parse(String(b.summary?.started_at ?? '')) || 0
    return bStart - aStart
  })

  return output
}

const defaultExperimentsPlugin = {
  name: 'default-experiments-api',
  configureServer(server: { middlewares: { use: Function } }) {
    server.middlewares.use(async (req: { method?: string; url?: string }, res: { setHeader: Function; end: Function }, next: Function) => {
      const pathname = (req.url ?? '').split('?')[0]
      if (req.method !== 'GET' || pathname !== '/api/experiments/default') {
        next()
        return
      }

      try {
        const runs = await loadDefaultRuns()
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ runs }))
      } catch (error) {
        res.setHeader('Content-Type', 'application/json')
        const message = error instanceof Error ? error.message : 'Unable to load default experiments'
        res.end(JSON.stringify({ runs: [], error: message }))
      }
    })
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), defaultExperimentsPlugin],
})
