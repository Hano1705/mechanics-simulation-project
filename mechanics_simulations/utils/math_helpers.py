import numpy as np

def calculate_distance(point1: np.ndarray, point2: np.ndarray):

    return np.sqrt( np.sum( np.square(point1 - point2) ) )