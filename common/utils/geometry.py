import numpy as np
from typing import List

def rotateVector(vector, angle):
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, vector)

def angleBetweenVectors(v1, v2):
    u = np.array(v1)
    v = np.array(v2)
    dot_product = np.dot(u, v)

    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle = np.arccos(dot_product)

    return angle  # in radians

def normalizeVector(vector):
    return vector / np.linalg.norm(vector)

def distanceBetweenPoints(p1: List[float], p2: List[float]):
    return np.linalg.norm(np.array(p1) - np.array(p2))
