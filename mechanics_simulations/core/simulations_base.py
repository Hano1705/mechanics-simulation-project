# -----------
# A base class for simulations
# -----------

import numpy as np
from abc import ABC, abstractmethod
from codetiming import Timer

class Simulation(ABC):
    '''
        A simulator base class
    '''
    def __init__(self):
        pass
    
    @Timer(name='decorator', text= "Simulation elapsed time: {:.4f} seconds")
    def run_simulation(self, simulation_time: float|int, timestep: float):
        '''Runs simulation'''
        # check for argument validity
        if simulation_time <= 0:
            raise ValueError(f'simulation_time must be greater than 0: {simulation_time}')
        if timestep <= 0:
            raise ValueError(f'timestep must be greater than 0: {simulation_time}')
        if simulation_time < timestep:
            raise ValueError(f'timestep ({timestep}) must be less than simulation_time ({simulation_time})')

        # checks for error in simulator method outputs
        self._runtime_error_checker()

        # local variables for simulation
        time_list = [0.0]
        state_list = [self._get_initial_state()]

        print("Running simulation")
        while time_list[-1] < simulation_time:
            # update time and state
            state = self._propagate_once(state=state_list[-1], timestep=timestep)
            # append results to result lists
            time_list.append(time_list[-1]+timestep)
            state_list.append(state)
        
        print("Finished simulation")
        self.time = np.array(time_list)
        self.state = np.array(state_list)

        return self.time, self.state
    
    @abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        '''returns initial state of system. Implemented in subclass'''
        pass

    @abstractmethod
    def _propagate_once(self, state: np.ndarray, timestep: float | int)-> np.ndarray:
        pass

    def _runtime_error_checker(self):
        '''
            checks for errors in the returned objects of the three abstract methods. 
        '''
        # ensure correct state and derivative types and shapes
        if not isinstance(self._get_initial_state(), np.ndarray):
            raise TypeError(f'._get_initial_state must return {np.ndarray}, but returned {type(self._get_initial_state())}.')
        
        # ensure correct propagated state type and shape
        mock_timestep = 0.01
        if not isinstance(self._propagate_once(self._get_initial_state(), mock_timestep), np.ndarray):
            raise TypeError(f'_propagate_once must return {np.ndarray}, but returned {type(self._propagate_once(self._get_initial_state(), mock_timestep))}')
        if not self._propagate_once(self._get_initial_state(), mock_timestep).shape == self._get_initial_state().shape:
            raise ValueError(f'._propagate_once and ._get_initial_state must return arrays of equivalent dimension.')
        
        return None