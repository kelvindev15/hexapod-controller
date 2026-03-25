from typing import Dict, List, Optional, Any
import json
import numpy as np
from numpy.typing import NDArray

# File I/O operations
def readSystemInstruction() -> str:
    """Read system instructions from file."""
    with open('system_instruction.txt', 'r') as file:
        return file.read()

def readUserPrompt() -> str:
    """Read user prompt from file."""
    with open('user_prompt.txt', 'r') as file:
        return file.read()

def save_robot_pose(robot: Any) -> None:
    """Save robot pose to JSON file."""
    with open('robot_pose.json', 'w') as file:
        json.dump(robot.get_pose(), file)

def read_robot_pose() -> Dict[str, Any]:
    """Read robot pose from JSON file."""
    with open('robot_pose.json', 'r') as file:
        return json.load(file)

def read_target_position() -> Dict[str, Any]:
    """Read target position from JSON file."""
    with open('target_position.json', 'r') as file:
        return json.load(file)

# LIDAR data processing
class LidarSection:
    def __init__(self, min_angle: float, max_angle: float, readings: NDArray):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.readings = readings
        self.min_distance = float(np.min(readings)) if readings.size > 0 else None
        self.min_distance_angle = (
            min_angle + (np.argmin(readings) / max(len(readings)-1, 1)) * (max_angle - min_angle)
            if readings.size > 0 else None
        )

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "minAngle": self.min_angle,
            "maxAngle": self.max_angle,
            "min_distance": self.min_distance,
            "min_distance_angle": self.min_distance_angle
        }

def getDistancesFromLidar(readings: List[float], fov_degrees: int, num_sections: int = 3) -> Dict[str, Any]:
    """
    Process LIDAR readings and return structured distance data divided into num_sections equally.
    
    Args:
        readings: Raw LIDAR distance readings
        fov_degrees: Field of view in degrees
        num_sections: Number of equal angular sections to split the FOV into
    
    Returns:
        Dictionary containing processed LIDAR data: front_distance and a list of section dicts
    """
    readings_array = np.array(readings)
    if len(readings_array) == 0 or num_sections < 1:
        return {"front_distance": None, "sections": []}

    section_angle = float(fov_degrees) / float(num_sections)
    points_per_degree = len(readings_array) / float(fov_degrees)

    # compute split indices to cover entire array
    points_per_section = [int(round(points_per_degree * section_angle)) for _ in range(num_sections)]
    # adjust to ensure sum matches length by distributing difference
    total_assigned = sum(points_per_section)
    diff = len(readings_array) - total_assigned
    i = 0
    while diff != 0:
        points_per_section[i % num_sections] += 1 if diff > 0 else -1
        diff = len(readings_array) - sum(points_per_section)
        i += 1

    sections = []
    start_idx = 0
    fov_start = -float(fov_degrees) / 2.0
    for i, pts in enumerate(points_per_section):
        end_idx = start_idx + max(0, pts)
        min_angle = fov_start + i * section_angle
        max_angle = min_angle + section_angle
        slice_readings = readings_array[start_idx:end_idx] if end_idx > start_idx else np.array([])
        sections.append(
            LidarSection(min_angle, max_angle, slice_readings).to_dict()
        )
        start_idx = end_idx

    # front distance: mean of central window
    half_width = 3
    n = len(readings_array)
    if n > 2 * half_width:
        center = n // 2
        front_slice = readings_array[max(0, center - half_width): min(n, center + half_width + 1)]
        front_distance = float(front_slice.mean()) if front_slice.size > 0 else None
    else:
        front_distance = None

    return {"front_distance": front_distance, "sections": sections}


def format_distance_reading(value: Optional[float], precision: int = 2) -> str:
    """Format a distance reading with specified precision."""
    return f"{round(value, precision)}" if value is not None else "N/A"


def getDistanceDescription(distances: Dict[str, Any]) -> str:
    """Generate a human-readable description of LIDAR distances for an arbitrary number of sections."""
    lines = []
    lines.append(f"Lidar distances:")
    lines.append(f"  - Front: {format_distance_reading(distances.get('front_distance'))}")

    sections = distances.get("sections", [])
    for idx, sec in enumerate(sections):
        min_a = sec.get("minAngle")
        max_a = sec.get("maxAngle")
        dist = sec.get("min_distance")
        angle = sec.get("min_distance_angle")
        angle_str = f"{np.round(angle)}" if angle is not None else "N/A"
        lines.append(
            f"  - Section {idx+1} ([{min_a}, {max_a}] degrees): {format_distance_reading(dist)} at angle {angle_str} degrees"
        )

    return "\n".join(lines)
