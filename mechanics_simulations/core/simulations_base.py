# -----------
# A base class for simulations
# -----------

import numpy as np
from abc import ABC, abstractmethod
from codetiming import Timer

class Simulation(ABC):
    '''
        A simulator class
    '''
    def __init__(self, propagator):
        
        # initializing the simulation
        self._propagator = propagator
    
    @Timer(name='decorator', text= "Simulation elapsed time: {:.4f} seconds")
    def run_simulation(self, simulation_time: float|int, timestep: float):
        '''Runs simulation'''
        
        # check input arguments
        if simulation_time <= 0:
            raise ValueError(f'simulation_time must be greater than 0: {simulation_time}')
        if timestep <= 0:
            raise ValueError(f'timestep must be greater than 0: {simulation_time}')
        if simulation_time < timestep:
            raise ValueError(f'timestep ({timestep}) must be less than simulation_time ({simulation_time})')

        # ensure correct state and derivative types and shapes
        if not isinstance(self._get_initial_state(), np.ndarray):
            raise TypeError(f'._get_initial_state must return {np.ndarray}, but returned {type(self._get_initial_state())}.')
        if not isinstance(self._compute_derivatives(self._get_initial_state()), np.ndarray):
            raise TypeError(f'._compute_derivatives must return {np.ndarray}, but returned {type(self._compute_derivatives(self._get_initial_state()))}.')
        if not self._get_initial_state().shape == self._compute_derivatives(self._get_initial_state()).shape:
            raise ValueError(f'._compute_derivatives and ._get_initial_state must return arrays of equivalent dimension.')
        
        # local variables for simulation
        time_list = [0]
        state_list = [self._get_initial_state()]

        print("Running simulation")
        while time_list[-1] < simulation_time:
            # update time and state
            time, state = self._propagator(rhs_func=self._compute_derivatives
                                      , time=time_list[-1]
                                      , state=state_list[-1]
                                      , timestep=timestep)
            # append results to result lists
            time_list.append(time)
            state_list.append(state)
        
        print("Finished simulation")
        self.time = np.array(time_list)
        self.state = np.array(state_list)

        return time, state
    
    @abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        '''returns initial state of system. Implemented in subclass'''
        pass

    @abstractmethod
    def _compute_derivatives(self, state: np.ndarray) -> np.ndarray:
        '''returns derivatives of the current state. Implemented in subclass'''
        pass