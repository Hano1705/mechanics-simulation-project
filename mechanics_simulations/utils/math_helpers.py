import numpy as np

def calculate_distance(point1: np.ndarray, point2: np.ndarray):

    return np.sqrt( np.sum( np.square(point1 - point2) ) )

def calculate_distances(points1: np.ndarray, points2: np.ndarray):
    
    return np.sqrt( np.sum(np.square(points1 - points2), axis=1) )