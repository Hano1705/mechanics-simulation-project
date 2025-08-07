import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import numpy as np

from matplotlib import animation

from mechanics_simulations import Animation

from mechanics_simulations.double_pendulum.pendulum_objects import Pendulum
from mechanics_simulations.double_pendulum.pendulum_objects import DoublePendulum
from mechanics_simulations.double_pendulum.double_pendulum_simulations import PendulumSimulation
from mechanics_simulations.double_pendulum.double_pendulum_simulations import DoublePendulumSimulation

class PendulumAnimation(Animation):
    '''
        A class for pendulum animations.
    '''
    def __init__(self, simulation: PendulumSimulation):
        super().__init__(simulation=simulation)
        self._simulation = simulation
    
    def initialize_animation(self):
        '''
            Initializes the animation of the pendulum
        '''
        # simulation time
        self._t = self._simulation.time

        # calculate and set cartesian coordinates
        self._simulation.calculate_cartesian_coordinates()
        self._x, self._y = self._simulation.cartesian_coord.transpose()

        self._x0, self._y0 = self._simulation.object.origin
        
        # set axis limites
        _length = self._simulation.object.length
        _width = _length * 1.5

        self._ax.set_xlim(left = self._x0 - _width, right = self._x0 + _width)
        self._ax.set_ylim(bottom = self._y0 - _width, top = self._y0 + _width)

        # axis labels
        self._ax.set_xlabel('x (m)')
        self._ax.set_ylabel('y (m)')

        # set ax as square
        self._ax.set_aspect('equal', adjustable='box')

        # define artists for pendulum
        self._pendulum, = self._ax.plot([self._x0, self._x[0]]
                                        , [self._y0, self._y[0]]
                                        ,'o-', color='blue')
        
        self._trace, = self._ax.plot(self._x[0], self._y[0]
                                     , color='red', alpha=0.3)

    def _update_frame(self, frame):
        '''
            updates the frames for the animation

            Parameters:
            -----------
            frame: the present frame
        '''
        # update the pendulum plot
        self._pendulum.set_xdata(np.array([self._x0, self._x[frame]]))
        self._pendulum.set_ydata(np.array([self._y0, self._y[frame]]))

        # update trace plot
        if frame>100:
            self._trace.set_xdata(self._x[frame-100:frame])
            self._trace.set_ydata(self._y[frame-100:frame])
        else:
            self._trace.set_xdata(self._x[:frame])
            self._trace.set_ydata(self._y[:frame])

        return (self._pendulum, self._trace)

class DoublePendulumAnimation(Animation):
    '''
        A class for double pendulum animations.
    '''
    def __init__(self, simulation: DoublePendulumSimulation):
        super().__init__(simulation=simulation)
        self._simulation = simulation

    def initialize_animation(self):
        '''
            Initializes the animation of the projectile
        '''
        # simulation time
        self._t = self._simulation.time
        # calculate and set cartesian coordinates
        self._simulation.calculate_cartesian_coordinates()
        self._x1, self._y1, self._x2, self._y2 = self._simulation.cartesian_coord.transpose()

        # set axis limits
        self._x0, self._y0 = self._simulation.object.origin
        _width = (self._simulation.object.pendulum1.length 
                 + self._simulation.object.pendulum2.length) * 1.5
        
        self._ax.set_xlim(left = self._x0 - _width, right = self._x0 + _width)
        self._ax.set_ylim(bottom = self._y0 - _width, top = self._y0 + _width)

        # axis labels
        self._ax.set_xlabel('x (m)')
        self._ax.set_ylabel('y (m)')

        # set ax as square
        self._ax.set_aspect('equal', adjustable='box')

        # define artists
        self._pendulum_artist, = self._ax.plot(
                                        [self._x0, self._x1[0], self._x2[0]]
                                        , [self._y0, self._y1[0], self._y2[0]]
                                        , 'o-', color='blue')
        
        self._trace_artist, = self._ax.plot(self._x2[0], self._y2[0]
                                            , color='red',alpha=0.3)
        
        self._text_artist = self._ax.text(0.1, 0.9, s=f"t = {self._t[0]:.1f} s"
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
        self._pendulum_artist.set_xdata([self._x0, self._x1[frame]
                                         , self._x2[frame]])
        self._pendulum_artist.set_ydata([self._y0, self._y1[frame]
                                         , self._y2[frame]])
                                

        # update trace plot
        tail_length = 100
        if frame>tail_length:
            self._trace_artist.set_xdata(self._x2[frame-tail_length:frame])
            self._trace_artist.set_ydata(self._y2[frame-tail_length:frame])
        else:
            self._trace_artist.set_xdata(self._x2[:frame])
            self._trace_artist.set_ydata(self._y2[:frame])

        # update text
        self._text_artist.set_text(s=f"t = {self._t[frame]:.1f} s")

        return (self._pendulum_artist, self._trace_artist)
    