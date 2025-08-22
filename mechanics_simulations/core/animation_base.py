import matplotlib.pyplot as plt
from matplotlib import animation
from abc import ABC, abstractmethod

class Animation(ABC):
    '''An abstract base class for animation subclasses'''

    def __init__(self, simulation):
        '''initialization of base class object.'''
        self._simulation = simulation
        self._fig, self._ax = plt.subplots()
        self._frames = len(simulation.time)

    def show_animation(self, interval_frames: float, repeat_delay: float):
        '''
            shows animation.

            Parameters:
            -----------
            interval_frames: minimum time delay between frames (unit: ms)
            repeat_delay: time between succesive repeats (unit: ms)    
        '''
        # initialize animation
        self._initialize_animation()

        # instantiate animation
        ani = animation.FuncAnimation(fig=self._fig 
                                    , func=self._update_frame # type: ignore
                                    , frames=self._frames
                                    , interval=interval_frames
                                    , repeat_delay=repeat_delay)
        plt.show()

    @abstractmethod
    def _initialize_animation(self):
        '''initializes animation, implemented in subclass'''
        pass

    @abstractmethod
    def _update_frame(self):
        '''updates frame of the animation, implemented in subclass'''
        pass