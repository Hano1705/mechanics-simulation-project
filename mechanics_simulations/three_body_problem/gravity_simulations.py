import numpy as np
from numba import jit
from functools import partial

from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravity_object import GravitationalObject, CelestialSystem
from mechanics_simulations.utils.math_helpers import calculate_distance
# -----------------------------------------------------------------------    
class NBodySimulation(Simulation):
    '''
        N-body simulation class. Takes a dictionary of GravitationalObjects and a propagator,
        and runs the evolution of the system of GravitationalObjects for a given amount of time.
    '''
    def __init__(self, system: CelestialSystem):
        '''
            Initialize the N-body simulation.

            Parameters:
            system: CelstialSystem object
        '''
        # set default propagator for the NBody simulation
        self._set_default_propagator()

        # define variables for the physical parameters
        self.system = system.get_system()
        self.masses = np.fromiter((cel_object.mass for cel_object in self.system.values()), dtype=np.float32)
        self.gravitational_constant = system.gravitational_constant

    def _get_initial_state(self) -> np.ndarray:
        '''
            returns the initial state of the system as a np.array with dimension
            (N, 2, 2) corresponding to N objects, 2 types of properties (positon/velocity)
            and 2 coordinates.
        '''
        _temp = [[cel_object.position, cel_object.velocity] for cel_object in self.system.values()]

        return np.array(_temp, dtype=np.float32) # object, property, coordinate
    
    def _propagate_once(self, state: np.ndarray, timestep: float | int) -> np.ndarray:
        '''
            calculates the state of the system at the next timestep, given the timestep and the current step
        '''
        return self._propagator(state=state, timestep=timestep) # type:ignore

    def _set_default_propagator(self):
        '''
            Sets the default propagator for the simulation, which incorporates the _compute_derivatives method
        '''
        propagator=RK4Integrator().propagate_state
        self._propagator = partial(propagator, rhs_func=self._compute_derivatives) # type:ignore
        return None

    def _compute_derivatives(self, state: np.ndarray) -> np.ndarray:
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

    def set_propagator(self, propagator):
        '''
            Sets the user defined propagator for the simulation.

            parameters:
            propagator: the chosen propagator for the simulation.
        '''
        self._propagator = propagator
        return None
    
    def _runtime_propagator_checker(self):
        
        if not isinstance(self._compute_derivatives(self._get_initial_state()), np.ndarray):
            raise TypeError(f'._compute_derivatives must return {np.ndarray}, but returned {type(self._compute_derivatives(self._get_initial_state()))}.')
        if not self._get_initial_state().shape == self._compute_derivatives(self._get_initial_state()).shape:
            raise ValueError(f'._compute_derivatives and ._get_initial_state must return arrays of equivalent dimension.')
        
        return None
    