# --------
# A module for defining graviational objects
# --------

class GravitationalObject():
    '''A single graviational object'''

    def __init__(self, mass: float, position: list[float], velocity: list[float]):
        self.mass = mass
        self.position = position # [x, y]
        self.velocity = velocity # [vx, vy]




