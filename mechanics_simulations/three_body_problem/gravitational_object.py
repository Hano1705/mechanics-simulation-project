# --------
# A module for defining graviational objects
# --------
import numpy as np
from astropy.constants import M_sun, M_earth # type:ignore

class GravitationalObject():
    '''A single graviational object'''

    def __init__(self, mass: float, position: list[float], velocity: list[float]):
        self.mass = mass
        self.position = position # [x, y]
        self.velocity = velocity # [vx, vy]


class CelestialSystem():

    def __init__(self, name, celestial_objects: dict[str, GravitationalObject]|None = None):

        self.name = name
        self.celestial_objects = {} if celestial_objects is None else celestial_objects

    def add_object(self, name, object):

        self.celestial_objects[name] = object

    def remove_object(self, name):

        self.celestial_objects.pop(name, None)

    def get_object(self,name):

        return self.celestial_objects[name]
    
    def get_system(self):

        return self.celestial_objects

    @classmethod
    def solar_system(cls):
        return cls(name='Solar System', celestial_objects={
            'Sun': GravitationalObject(mass=1, position=[0, 0], velocity=[0, 0])
            , 'Mercury': GravitationalObject(mass=0.0553*M_earth/M_sun, position=[0.39, 0], velocity=[0, 0.39*2*np.pi/0.240846])
            , 'Venus': GravitationalObject(mass=0.815*M_earth/M_sun, position=[0.72, 0], velocity=[0, 0.72*2*np.pi/0.615])
            , 'Earth': GravitationalObject(mass=M_earth/M_sun, position=[1, 0], velocity=[0, 2*np.pi])
            , 'Mars': GravitationalObject(mass=0.1075*M_earth/M_sun, position=[1.52, 0], velocity=[0, 1.52*2*np.pi/1.881])
            , 'Jupiter': GravitationalObject(mass=317.8*M_earth/M_sun, position=[5.2, 0], velocity=[0, 5.2*2*np.pi/11.86])
        })
        print('Created the solar system')
    
    @classmethod
    def chaotic_three_body(cls):
        object1 = GravitationalObject(mass=1, position=[0,0], velocity=[-12*0.5,-12*0.5])
        object2 = GravitationalObject(mass=1, position=[-1.4,0], velocity=[3*0.53,6*0.35])
        object3 = GravitationalObject(mass=1, position=[1.4,0], velocity=[3*0.35,3*0.53])

        return cls(name='Chaotic three body system', celestial_objects={'object1': object1,'object2': object2,'object3': object3})
