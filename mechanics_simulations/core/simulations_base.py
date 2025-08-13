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
        # local variables for simulation
        time_list = [0]
        state_list = [self.get_initial_state()]

        print("Running simulation")
        while time_list[-1] < simulation_time:
            # update time and state
            time, state = self._propagator(rhs_func=self.compute_derivatives
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
    def get_initial_state(self):
        '''returns initial state of system. Implemented in subclass'''
        pass

    @abstractmethod
    def compute_derivatives(self):
        '''returns derivatives of the current state. Implemented in subclass'''
        pass