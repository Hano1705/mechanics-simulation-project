# --------------
# A set of integrator classes
# --------------
import numpy as np
from types import FunctionType

class RK4Integrator():
    '''
        An RK4 integrator class, used to integrate states (can be though of as vector functions), given derivative
        of state (each entry in the vector function).
    '''
    def integrate_state(self, derivative_func: FunctionType, state: np.ndarray, timestep: float) -> tuple:
        '''
            Integrates a given state one timestep, using input derivative function. The derivative function must return
            a np.ndarray of same shape as state.
            
            Parameters:
            derivative_func: a function giving the time derivative of the state variables.
            state: current state of the object to be propagated.
            timestep: timestep of the integration.

            Returns:
            tuple: np.ndarray of shape equivalent to state.
        '''
        # calculate RK4 protocol variables
        k1 = derivative_func(state)
        k2 = derivative_func(state + timestep * k1/2)
        k3 = derivative_func(state + timestep * k2/2)
        k4 = derivative_func(state + timestep * k3)

        # return new state
        return state + timestep/6 * (k1 + 2*k2 + 2*k3 + k4)