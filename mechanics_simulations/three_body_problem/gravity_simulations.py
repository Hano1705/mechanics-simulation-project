import numpy as np
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
            Initialize the N-body simulation

            Parameters:
            objects: list of gravitational objects in simulation
            propagator: integrator
            scalefactor_G: a refactoring scale, to choose units in the simulation. Default: G in SI-units.
        '''

        super().__init__(propagator=propagator)

        self.system = system

        _temp_list= []
        for name, cel_object in self.system.items():
            _temp_list.append(cel_object.mass)
        self.masses = np.array(_temp_list, dtype=np.float32)

        self.gravitational_constant = (4 * np.pi**2)

    def get_initial_state(self):

        _temp_list = []
        for name, cel_object in self.system.items():
            _temp_list.append([cel_object.position, cel_object.velocity])

        return np.array(_temp_list, dtype=np.float32) # object, pos, vel
    
    def compute_derivatives(self, state):
        
        positions, velocities = np.transpose(state, (2,0,1))
        
        displacements = positions[:, np.newaxis] - positions
        mass_weighted_displacements = self.masses[:,np.newaxis] * displacements

        mask = ~np.eye(displacements.shape[0],dtype=bool)[:,:,np.newaxis] * np.ones(displacements.shape, dtype=bool)
        mass_weighted_displacements = mass_weighted_displacements[mask].reshape((displacements.shape[0],displacements.shape[1]-1,displacements.shape[2]))
        displacements = displacements[mask].reshape((displacements.shape[0],displacements.shape[1]-1,displacements.shape[2]))
        
        distances = np.sqrt( np.sum( np.square(displacements), axis=2))
        


        temp = - self.gravitational_constant * (1 / distances **3)[:,:,np.newaxis] * mass_weighted_displacements
        temp = np.sum(temp, axis=1)


        pos_derivatives = velocities
        vel_derivatives = temp
        return np.transpose(np.array([pos_derivatives,vel_derivatives]), (1,2,0))
            

system = CelestialSystem.solar_system()
RK4solver = RK4Integrator().propagate_state
sim = NBodySimulation(system=system.celestial_objects, propagator=RK4solver)
sim.run_simulation(simulation_time=2, timestep=0.01)
print('breakpoint')