import numpy as np

from mechanics_simulations.core.integrators import RK4Integrator
from mechanics_simulations.double_pendulum.pendulum_objects import Pendulum
from mechanics_simulations.double_pendulum.pendulum_objects import DoublePendulum
from mechanics_simulations.double_pendulum.double_pendulum_simulations import PendulumSimulation
from mechanics_simulations.double_pendulum.double_pendulum_simulations import DoublePendulumSimulation
from mechanics_simulations.double_pendulum.double_pendulum_animations import PendulumAnimation
from mechanics_simulations.double_pendulum.double_pendulum_animations import DoublePendulumAnimation

def _animate_pendulum():
    # instantiate the pendulum
    pendulum = Pendulum(mass=1, length=1, origin=[0,0])
    pendulum.set_pendulum(theta=np.pi/2, w=0)
    print("pendulum instantiated")
    
    rk_solver = RK4Integrator()

    my_simulation = PendulumSimulation(pendulum=pendulum, propagator=rk_solver.propagate_state)
    my_simulation.run_simulation(simulation_time=10, timestep=0.01)
    print('finished simulation')
    my_animation = PendulumAnimation(simulation = my_simulation)
    my_animation.show_animation(interval_frames=20, repeat_delay=1000)
    print('finished animation')

def _animate_double_pendulum():
    # instantiate the two pendula making up the double pendulum
    pendulum1 = Pendulum(mass=1, length=1, origin=[0,0])
    pendulum2 = Pendulum(mass=1, length=0.8)
    # instantiate the double pendulum
    double_pendulum = DoublePendulum(pendulum1=pendulum1, pendulum2=pendulum2)
    double_pendulum.set_double_pendulum(theta1=np.pi/4, w1=0,
                                       theta2=-np.pi/4, w2=0)
    print("double pendulum instantiated")
    
    rk_solver = RK4Integrator()

    my_simulation = DoublePendulumSimulation(double_pendulum=double_pendulum, propagator=rk_solver.propagate_state)
    my_simulation.run_simulation(simulation_time=100, timestep=0.01)
    my_simulation.calculate_cartesian_coordinates()
    print('finished simulation')
    my_animation = DoublePendulumAnimation(simulation = my_simulation)
    my_animation.show_animation(interval_frames=20, repeat_delay=1000)
    print('finished animation')

if __name__ == '__main__':
    _animate_double_pendulum()