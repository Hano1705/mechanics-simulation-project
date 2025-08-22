import unittest
import numpy as np
from astropy.constants import M_sun, M_earth # type:ignore

from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravity_object import GravitationalObject, CelestialSystem
from mechanics_simulations.three_body_problem.gravity_simulations import  NBodySimulation

class TestGravitySimulationInit(unittest.TestCase):
    '''
        tests constructor, that masses of input system are interpreted correctly by simulator,
        and that the initial positions are interpreted correct
    '''
    propagator = RK4Integrator()
    test_system = CelestialSystem.solar_system()
    test_sim = NBodySimulation(system=test_system.get_system(), propagator=propagator.propagate_state)

    def test_run_simulation_invalid_arguments(self):
        '''Tests exceptions raised with invalid input arguments to .run_simulation() method'''
        with self.assertRaises(ValueError):
            self.test_sim.run_simulation(simulation_time=-1, timestep=0.01)
        with self.assertRaises(ValueError):
            self.test_sim.run_simulation(simulation_time=1, timestep=-0.01)
        with self.assertRaises(ValueError):
            self.test_sim.run_simulation(simulation_time=1, timestep=2)

    def test_constructor(self):
        '''Tests that types for system and masses are correct after instantiation'''
        self.assertIs(type(self.test_sim.system), dict)
        self.assertIs(type(self.test_sim.masses), np.ndarray)

    def test_system_masses(self):
        '''Tests that the simulation obtains the correct masses from the "system" input'''
        target_masses = np.array([1, 0.0553*M_earth/M_sun, 0.815*M_earth/M_sun, M_earth/M_sun
                                  , 0.1075*M_earth/M_sun, 317.8*M_earth/M_sun]
                                 , dtype=np.float32)

        self.assertTrue((self.test_sim.masses == target_masses).all())

    def test_get_initial_state(self):
        '''Tests that the simulation obtains the correct initial state from the "system" input'''
        target_initial_state = ([[0, 0], [0, 0]]
                                ,[[0.39, 0], [0, 0.39*2*np.pi/0.240846]]
                                ,[[0.72, 0], [0, 0.72*2*np.pi/0.615]]
                                ,[[1, 0], [0, 2*np.pi]]
                                ,[[1.52, 0], [0, 1.52*2*np.pi/1.881]]
                                ,[[5.2, 0], [0, 5.2*2*np.pi/11.86]]
                                )
        target_initial_state = np.array(target_initial_state, dtype=np.float32)

        self.assertTrue((self.test_sim._get_initial_state() == target_initial_state).all())

class GravitySimulationsPhysicsTests(unittest.TestCase):

    def test_nbody_forces(self):
        '''
            Tests Newtonian force for two objects of different masses and different positions
        '''
        propagator = RK4Integrator()

        # testcases
        test_masses = ((1, 1),
                        (1, 1),
                        (0, 1),
                        (0, 0),
                        (1, 1, 1),
                        (1, 1, 1),
                        )
        test_positions = (([-0.5,0],[0.5,0]),
                          ([0,0],[0,1]),
                          ([0,0],[0,1]),
                          ([0,0],[0,1]),
                          ([0,0],[1,0],[2,0]),
                          ([0,0],[1,0],[0.5,np.sqrt(3)/2]),
                          )
        expected_forces = (([4*np.pi**2, 0],[-4*np.pi**2, 0]),
                           ([0, 4*np.pi**2],[0,-4*np.pi**2]),
                           ([0, 4*np.pi**2],[0,0]),
                           ([0,0],[0,0]),
                           ([4*np.pi**2*5/4, 0], [0, 0], [-4*np.pi**2*5/4, 0]),
                           ([4*np.pi**2*3/2, 4*np.pi**2*np.sqrt(3)/2], [-4*np.pi**2*3/2, 4*np.pi**2*np.sqrt(3)/2], [0, -4*np.pi**2*np.sqrt(3)]),
                           )

        # run testcases
        for masses, positions, forces in zip(test_masses, test_positions, expected_forces):
            
            test_system = CelestialSystem(name='foo', celestial_objects={})

            # construct system
            for it, (mass, position, force) in enumerate(zip(masses, positions, forces)):
                object = GravitationalObject(mass=mass, position=position, velocity=[0,0])
                name = f'object{it}'
                test_system.add_object(name=name, object=object)

            test_sim = NBodySimulation(system=test_system.get_system(), propagator=propagator.propagate_state)

            # test case
            expected_result = np.array(forces, dtype=np.float32)
            calculated_result = test_sim._compute_derivatives(state=test_sim._get_initial_state())[:,1,:]
            self.assertTrue((expected_result == calculated_result).all())



if __name__ == '__main__':
    unittest.main(verbosity=1)