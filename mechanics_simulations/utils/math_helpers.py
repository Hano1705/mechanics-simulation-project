import numpy as np

def calculate_distance(point1: np.ndarray, point2: np.ndarray):

    return np.sqrt( np.sum( np.square(point1 - point2) ) )

def calculate_distances(points1: np.ndarray, points2: np.ndarray):
    
    return np.sqrt( np.sum(np.square(points1 - points2), axis=1) )

def _fat_helper(positions, velocities, masses):
    # calculate r between objects
    displacements = positions - positions[:, np.newaxis]
    # calculate mass weighted r between objects
    mass_weighted_displacements = masses[:,np.newaxis] * displacements
    
    # create mask to remove "diagonal" entries: (N,N,2) -> (N,N-1,2)
    shp = displacements.shape
    mask = ~np.eye(shp[0],dtype=bool)[:,:,np.newaxis] * np.ones(shp, dtype=bool)
    mass_weighted_displacements = mass_weighted_displacements[mask].reshape((shp[0],shp[1]-1,shp[2]))
    displacements = displacements[mask].reshape((shp[0],shp[1]-1,shp[2]))
    
    # calculate distances between objects
    distances = np.sqrt( np.sum( np.square(displacements), axis=2))

    # calculate F/m contribution from each object to each object
    force_over_mass = (4 * np.pi**2) * (1 / distances **3)[:,:,np.newaxis] * mass_weighted_displacements

    pos_derivatives = velocities
    vel_derivatives = np.sum(force_over_mass, axis=1)
    return np.transpose(np.array([pos_derivatives,vel_derivatives]), (1,0,2))