import numpy as np
import matplotlib as plt
import seaborn; seaborn.set_theme()

from mechanics_simulations import Animation
from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravity_simulations import NBodySimulation

from mechanics_simulations.three_body_problem.gravitational_object import GravitationalObject
from mechanics_simulations.three_body_problem.gravitational_object import CelestialSystem
# ----------------------------------------------------------------------------------------   
class NBodyAnimation(Animation):

    def __init__(self, simulation: NBodySimulation):
        super().__init__(simulation=simulation)
        self._simulation = simulation

    def _initialize_animation(self):
        '''
            Initializes the animation
        '''
        # simulation time
        self._t = self._simulation.time
        # coordinates
        self._positions, self._velocities = np.transpose(self._simulation.state, axes=[2,0,1,3])
        
        # set ax as square
        self._ax.set_aspect('equal', adjustable='box')

        # set axes labels
        self._ax.set_xlabel('x (AU)')
        self._ax.set_ylabel('y (AU)')

        # set axes
        center_of_mass = np.sum(self._simulation.masses[:,np.newaxis]*self._positions[0,:,:], axis=0) / np.sum(self._simulation.masses)
        self._ax.set_xlim(-6 + center_of_mass[0], 6 + center_of_mass[0])
        self._ax.set_ylim(-6 + center_of_mass[1], 6 + center_of_mass[1])

        self._point_artists = {}
        self._trace_artists = {}

        for it, name in enumerate(self._simulation.system.keys()):
            x_position = self._positions[0,it,0]
            y_position = self._positions[0,it,1]

            self._point_artists[name] = self._ax.scatter(x=x_position, y=y_position, label=name)     
            self._trace_artists[name], = self._ax.plot(x_position, y_position, alpha=0.4, linewidth=1)
        
        self._text_artist = self._ax.text(0.1, 0.9, s=f"t = {self._t[0]:.1f} yr"
                              , transform=self._ax.transAxes
                              , bbox={'facecolor':'green','alpha':0.2})
        self._ax.legend(bbox_to_anchor=(1.35, 1))
            

    def _update_frame(self, frame):
        for it, name in enumerate(self._simulation.system.keys()):
            self._point_artists[name].set_offsets(self._positions[frame, it,:])
            self._trace_artists[name].set_xdata(self._positions[:frame,it,0])
            self._trace_artists[name].set_ydata(self._positions[:frame,it,1])
        
        self._text_artist.set_text(s=f"t = {self._t[frame]:.1f} yr")

         # set axes
        origin = np.sum(self._simulation.masses[:,np.newaxis]*self._positions[frame,:,:], axis=0) / np.sum(self._simulation.masses)
        self._ax.set_xlim(-6 + origin[0], 6 + origin[0])
        self._ax.set_ylim(-6 + origin[1], 6 + origin[1])
            