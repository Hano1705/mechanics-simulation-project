import numpy as np
from astropy.constants import M_sun, M_earth, au, G # type: ignore

from mechanics_simulations import Animation
from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravitational_object import GravitationalObject
from mechanics_simulations.three_body_problem.gravitational_object import CelestialSystem

from mechanics_simulations.three_body_problem.gravity_simulations import NBodySimulation
from mechanics_simulations.three_body_problem.gravity_animations import NBodyAnimation

def run_solar_system_sim():
    solar_system = CelestialSystem.solar_system()

    integrator = RK4Integrator()

    sim = NBodySimulation(solar_system.get_system(), propagator = integrator.propagate_state)
    sim.run_simulation(simulation_time=5, timestep=0.01)

    anim = NBodyAnimation(simulation=sim)
    anim.show_animation(interval_frames=20, repeat_delay=1000)

# under development
def run_unstable_three_body_sim():
    object1 = GravitationalObject(mass=1, position=[0,0], velocity=[-12*0.5,-12*0.5])
    object2 = GravitationalObject(mass=1, position=[-1.4,0], velocity=[3*0.53,6*0.35])
    object3 = GravitationalObject(mass=1, position=[1.4,0], velocity=[3*0.35,3*0.53])

    figure_eight_system = CelestialSystem(name="Figure Eight", celestial_objects={
        'object1': object1
        ,'object2':object2
        ,'object3':object3
    })

    integrator = RK4Integrator()

    sim = NBodySimulation(figure_eight_system.get_system(), propagator = integrator.propagate_state)
    sim.run_simulation(simulation_time=20, timestep=0.01)

    anim = NBodyAnimation(simulation=sim)
    anim.show_animation(interval_frames=20, repeat_delay=1000)

if __name__== '__main__':
    # run_solar_system_sim()
    run_unstable_three_body_sim()