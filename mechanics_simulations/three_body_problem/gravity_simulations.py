import numpy as np
from numba import jit
from scipy.constants import G

from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravitational_object import GravitationalObject, CelestialSystem
from mechanics_simulations.utils.math_helpers import calculate_distance

class TwoBodySimulation(Simulation):

    def __init__(self, objects: list[GravitationalObject], propagator, scalefactor_G = 1):
        '''
            Initialize the two-body simulation

            Parameters:
            objects: list of gravitational objects in simulation
            propagator: integrator
            scalefactor_G: a refactoring scale, to choose units in the simulation. Default: G in SI-units.
        '''

        super().__init__(propagator=propagator)
        self.objects = objects

        self.properties = []
        for object in objects:
            self.properties.append(object.mass)

        self.gravitational_constant = scalefactor_G * G

    def get_initial_state(self):
        
        result = []
        for object in self.objects:
            result.append([object.position, object.velocity])
        
        return np.array(result) # (#objects, pos, vel)
    
    def compute_derivatives(self, state):
        
        (pos1, vel1), (pos2, vel2) = state
        mass1, mass2 = self.properties
        
        r = calculate_distance(pos1, pos2)
        r_vector = pos1 - pos2

        pos1_derivative = vel1
        vel1_derivative = - self.gravitational_constant * mass2 / r ** 3 * r_vector
        pos2_derivative = vel2
        vel2_derivative = - self.gravitational_constant * mass1 / r ** 3 * ( - r_vector )

        return np.array([[pos1_derivative,vel1_derivative], [pos2_derivative, vel2_derivative]])
    
class NBodySimulation(Simulation):

    def __init__(self, system: dict[str,GravitationalObject], propagator):
        '''
            Initialize the N-body simulation.

            Parameters:
            objects: list of gravitational objects in simulation
            propagator: integrator
        '''
        super().__init__(propagator=propagator)

        self.system = system

        # set .masses as numpy array of masses in the system
        _temp_list= []
        for name, cel_object in self.system.items():
            _temp_list.append(cel_object.mass)
        self.masses = np.array(_temp_list, dtype=np.float32)

        self.gravitational_constant = (4 * np.pi**2) # units of AU = 1, yr = 1

    def get_initial_state(self):
        '''
            returns the initial state of the system as a np.array with dimension
            (N, 2, 2) corresponding to N objects, 2 types of properties (positon/velocity)
            and 2 coordinates.

        '''
        _temp_list = []
        for name, cel_object in self.system.items():
            _temp_list.append([cel_object.position, cel_object.velocity])

        return np.array(_temp_list, dtype=np.float32) # object, property, coordinate
    

    def compute_derivatives(self, state: np.ndarray):
        '''
            computes the derivatives of the current state and returns it as np.array with
            dimension (N, 2, 2) corresponding to the get_initial_state method.
        '''
        # unpack position and velocity arrays
        positions, velocities = np.transpose(state, (1,0,2))
        
        # calculate r between objects
        displacements = positions - positions[:, np.newaxis]
        # calculate mass weighted r between objects
        mass_weighted_displacements = self.masses[:,np.newaxis] * displacements
        
        # create mask to remove "diagonal" entries: (N,N,2) -> (N,N-1,2)
        shp = displacements.shape
        mask = ~np.eye(shp[0],dtype=bool)[:,:,np.newaxis] * np.ones(shp, dtype=bool)
        mass_weighted_displacements = mass_weighted_displacements[mask].reshape((shp[0],shp[1]-1,shp[2]))
        displacements = displacements[mask].reshape((shp[0],shp[1]-1,shp[2]))
        
        # calculate distances between objects
        distances = np.sqrt( np.sum( np.square(displacements), axis=2))

        # calculate F/m contribution from each object to each object
        force_over_mass = self.gravitational_constant * (1 / distances **3)[:,:,np.newaxis] * mass_weighted_displacements

        pos_derivatives = velocities
        vel_derivatives = np.sum(force_over_mass, axis=1)

        # return numpy array in same format as original state
        return np.transpose(np.array([pos_derivatives,vel_derivatives]), (1,0,2))