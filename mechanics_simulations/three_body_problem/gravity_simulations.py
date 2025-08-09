import numpy as np
from scipy.constants import G

from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravitational_object import GravitationalObject
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
    
