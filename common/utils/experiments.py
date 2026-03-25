import os
import re
import json


def merge_plans(experiment_record: dict):
    merged_plan = {
        "goal": experiment_record.get("prompt", ""),
        "initialRobotPose": experiment_record.get("initialRobotPose", {}),
        "commands": []
    }
    for plan in experiment_record.get("plans", []):
        for command in plan.get("commands", []):
            if command["command"].upper() != "FEEDBACK":
                merged_plan["commands"].append(command)
    return merged_plan


def get_next_experiment_number(base_path='./experiments'):
    """
    Find the next available experiment number by scanning the experiments directory.

    Args:
        base_path (str): Path to the experiments directory

    Returns:
        int: Next available experiment number
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return 1

    experiment_dirs = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))
                       and d.startswith('experiment_')]

    if not experiment_dirs:
        return 1

    numbers = []
    for dir_name in experiment_dirs:
        match = re.match(r'experiment_(\d+)', dir_name)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers) + 1 if numbers else 1


def getExperimentFolderById(id):
    return f"experiments/experiment_{id}"


def replaySimulation(n: int, controller):
    plan = None
    with open(f'{getExperimentFolderById(n)}/experiment.json', 'r') as file:
        plan = json.loads(file.read())
    if plan:
        print("Replaying simulation")
        print(plan)
        controller.replay(merge_plans(plan))
        print("Plan execution terminated")
