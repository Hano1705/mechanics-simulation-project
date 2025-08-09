import numpy as np
import seaborn; seaborn.set_theme()

from mechanics_simulations import Animation
from mechanics_simulations.three_body_problem.gravity_simulations import TwoBodySimulation

from mechanics_simulations import Simulation
from mechanics_simulations import RK4Integrator

from mechanics_simulations.three_body_problem.gravitational_object import GravitationalObject

class TwoBodyAnimation(Animation):

    def __init__(self, simulation: TwoBodySimulation):
        super().__init__(simulation=simulation)
        self._simulation = simulation

    def initialize_animation(self):
        '''
            Initializes the animation of the projectile
        '''
        # simulation time
        self._t = self._simulation.time
        # coordinates
        ((self._x1,self._y1),_),((self._x2,self._y2),_) = np.transpose(self._simulation.state, axes=[1,2,3,0])

        self._ax.set_xlim(left = -1.5, right = 1.5)
        self._ax.set_ylim(bottom = -1.5, top = 1.5)

        # set ax as square
        self._ax.set_aspect('equal', adjustable='box')

        self._object1_artist = self._ax.scatter(self._x1[0],self._y1[0])
        self._object2_artist = self._ax.scatter(self._x2[0],self._y2[0])
        self._trace1_artist, = self._ax.plot(self._x1[0],self._y1[0], color='blue', alpha=0.4)
        self._trace2_artist, = self._ax.plot(self._x2[0],self._y2[0], color='orange', alpha=0.4)
        self._text_artist = self._ax.text(0.1, 0.9, s=f"t = {self._t[0]:.1f} yr"
                              , transform=self._ax.transAxes
                              , bbox={'facecolor':'green','alpha':0.2})

    def _update_frame(self, frame):
        '''
            updates the frames of the animation
            
            Parameters:
            -----------
            frame: the present frame
        '''
        # update the pendulum plot
        self._object1_artist.set_offsets(np.array([self._x1[frame], self._y1[frame]]))
        self._object2_artist.set_offsets(np.array([self._x2[frame], self._y2[frame]]))

        self._trace1_artist.set_xdata(self._x1[:frame])
        self._trace1_artist.set_ydata(self._y1[:frame])
        self._trace2_artist.set_xdata(self._x2[:frame])
        self._trace2_artist.set_ydata(self._y2[:frame])

        self._text_artist.set_text(s=f"t = {self._t[frame]:.1f} yr")

        return (self._object1_artist, self._object2_artist, self._trace1_artist, self._trace2_artist)
    
