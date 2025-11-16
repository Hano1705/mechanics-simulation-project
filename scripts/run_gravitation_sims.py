import numpy as np

from mechanics_simulations.n_body_problem.gravity_object import GravitationalObject, CelestialSystem
from mechanics_simulations.n_body_problem.gravity_simulations import NBodySimulation
from mechanics_simulations.n_body_problem.gravity_animations import NBodyAnimation

def run_solar_system_sim():
    solar_system = CelestialSystem.solar_system()

    sim = NBodySimulation(solar_system)
    sim.run_simulation(simulation_time=5, timestep=0.01)

    anim = NBodyAnimation(simulation=sim)
    anim.create_animation(interval_frames=20, repeat_delay=1000)

def run_chaotic_three_body_sim():
    three_body_system = CelestialSystem.chaotic_three_body()

    sim = NBodySimulation(three_body_system)
    sim.run_simulation(simulation_time=20, timestep=0.01)

    anim = NBodyAnimation(simulation=sim)
    anim.create_animation(interval_frames=20, repeat_delay=1000)

def run_figure_eight_sim(): 

    figure_eight_sys = CelestialSystem.figure_eight_system()

    sim = NBodySimulation(figure_eight_sys)

    # set the gravitational constant to 1
    sim.gravitational_constant = 1
    sim.run_simulation(simulation_time=10, timestep=0.05)

    anim = NBodyAnimation(simulation=sim)
    anim.create_animation(interval_frames=20, repeat_delay=1000)


if __name__== '__main__':
    run_solar_system_sim()
    # run_chaotic_three_body_sim()
    # run_figure_eight_sim()
    x = 1+1
