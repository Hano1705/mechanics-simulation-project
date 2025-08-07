import numpy as np

from mechanics_simulations.double_pendulum.pendulum_objects import Pendulum
from mechanics_simulations.double_pendulum.pendulum_objects import DoublePendulum

from mechanics_simulations.core.simulations_base import Simulation 
from mechanics_simulations.core.integrators import RungeKuttaIntegrator

class PendulumSimulation(Simulation):
    '''A subclass of the simulator class'''
    
    def __init__(self, object: Pendulum, propagator):
        super().__init__(object=object, propagator=propagator)
        self.object: Pendulum = object

    def get_initial_state(self):
        '''Returns the initial state of the double pendulum'''
        return self.object.state

    def compute_derivatives(self, state: np.ndarray):
        '''
            Computes the derivatives of the current state
        '''
        theta, w = state
        mass, length = self.object.get_properties()

        theta_derivative = w
        w_derivative = - 9.82 * np.sin(theta) / length

        return np.array([theta_derivative, w_derivative]
                         , dtype=np.float32)
    
    def calculate_cartesian_coordinates(self):

        theta, w = self.state.transpose()
        mass, length= self.object.get_properties()

        x = length * np.sin(theta) + self.object.origin[0]
        y = - length * np.cos(theta) + self.object.origin[1]

        self.cartesian_coord = np.stack([x,y], axis=1)
    
class DoublePendulumSimulation(Simulation):
    '''A subclass of the simulation class'''
    
    def __init__(self, object: DoublePendulum, propagator):
        super().__init__(object=object, propagator=propagator)
        self.object: DoublePendulum = object

    def get_initial_state(self):
        '''Returns the initial state of the double pendulum'''
        return self.object.state

    def compute_derivatives(self, state: np.ndarray):
        '''
            Computes the derivatives of the state variables
        
            Parameters:
            -----------
            state: current state of the object
        '''
        theta1, theta2, w1, w2 = state
        mass1, mass2, length1, length2 = self.object.get_properties()
        
        # calculate alphas
        alpha1 = (length2 / length1 * mass2 / (mass1 + mass2)
                   * np.cos(theta1-theta2) )
        alpha2 = length1 / length2 * np.cos(theta1-theta2)
        # calculate fs
        f1 = (- length2/length1 * mass2/(mass1+mass2) 
              * w1**2 * np.sin(theta1-theta2) 
              - 9.82/length1 * np.sin(theta1)
                )
        f2 = (length1/length2 * w1**2 * np.sin(theta1-theta2)
              - 9.82/length2 * np.sin(theta2)
                )
        # calculate gs
        g1 = (f1 - alpha1*f2) / (1-alpha1*alpha2)
        g2 = (f2 - alpha2*f1) / (1-alpha1*alpha2)

        # define derivative of state variables
        deriv_theta1 = w1
        deriv_theta2 = w2
        deriv_w1 = g1
        deriv_w2 = g2

        return np.array([deriv_theta1, deriv_theta2, deriv_w1, deriv_w2]
                        , dtype=np.float32)
    
    def calculate_cartesian_coordinates(self):

        theta1, theta2, w1, w2 = self.state.transpose()
        mass1, mass2, length1, length2 = self.object.get_properties()

        x1 = length1 * np.sin(theta1) + self.object.origin[0]
        y1 = - length1 * np.cos(theta1) + self.object.origin[1]
        x2 = x1 + length2 * np.sin(theta2)
        y2 = y1 - length2 * np.cos(theta2)

        self.cartesian_coord = np.stack([x1,y1,x2,y2], axis=1)
    