import numpy as np
from astropy.constants import M_sun, M_earth, au, G # type: ignore

from mechanics_simulations import Animation
from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravitational_object import GravitationalObject
from mechanics_simulations.three_body_problem.gravitational_object import CelestialSystem

from mechanics_simulations.three_body_problem.gravity_simulations import TwoBodySimulation
from mechanics_simulations.three_body_problem.gravity_simulations import NBodySimulation
from mechanics_simulations.three_body_problem.gravity_animations import TwoBodyAnimation
from mechanics_simulations.three_body_problem.gravity_animations import NBodyAnimation


def run_earth_sun_sim():
    sun = GravitationalObject(mass=1, position=[0,0], velocity=[0, 0])
    earth = GravitationalObject(mass=M_earth.value / M_sun.value, position=[1,0], velocity=[0, 2*np.pi])

    integrator = RK4Integrator()

    rescale_factor_units = (4 * np.pi**2) / G.value
    #rescale_factor_units = 1 / 4 * np.pi**2 / M_sun.value * au.value / (365.25 * 24 * 60 * 60)

    sim = TwoBodySimulation([earth, sun], propagator = integrator.propagate_state, scalefactor_G=rescale_factor_units)
    sim.run_simulation(simulation_time=1, timestep=0.001)
    print('done')

    anim = TwoBodyAnimation(simulation=sim)
    anim.show_animation(interval_frames=20, repeat_delay=1000)

# under development
def run_figure_eight_sim():
    object1 = GravitationalObject(mass=1, position=[0,0], velocity=[0, 0])
    object2 = GravitationalObject(mass=1, position=[1,0], velocity=[0, 0])
    object3 = GravitationalObject(mass=1, position=[0,1], velocity=[0, 0])

    figure_eight_system = CelestialSystem(name="Figure Eight", celestial_objects={
        'object1': object1
        ,'object2':object2
        ,'object3':object3
    })

    integrator = RK4Integrator()

    sim = NBodySimulation(figure_eight_system.get_system(), propagator = integrator.propagate_state)
    sim.run_simulation(simulation_time=1, timestep=0.001)
    print('done')

    anim = NBodyAnimation(simulation=sim)
    anim.show_animation(interval_frames=20, repeat_delay=1000)

if __name__== '__main__':
    run_figure_eight_sim()