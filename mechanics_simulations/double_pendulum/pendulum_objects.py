#######
#    This module defines the pendulum
#######
import numpy as np

class Pendulum():
    '''
        The pendulum class, with attributes corresponding to its physical features.
    '''
    def __init__(self, mass: float|int = 1
                 , length: float|int = 1
                 , origin: np.ndarray|list = np.array([0,0], dtype=np.float32)):

        self.mass = mass
        self.length = length
        if origin is list:
            self.origin = np.array(origin, dtype=np.float32)
        else:
            self.origin = origin

    def set_pendulum(self, theta: float|int, w: float|int):
        '''
            Set pendulum angle, also sets cartesian coordinates.
            -----------------------
            Parameters:
            theta: pendulum angle, starting from negative y-axis.
            w: angular velocity.
        '''
        self.theta = theta
        self.w = w
        self.state = [self.theta, self.w]

    def get_cartesian_state(self):
        self.x = self.length * np.sin(self.theta) + self.origin[0]
        self.y = - self.length * np.cos(self.theta) + self.origin[1]
        
        self.vx = self.length * self.w * np.cos(self.theta)
        self.vy = self.length * self.w * np.sin(self.theta)

        return self.x, self.y, self.vx, self.vy
       
    def get_properties(self):
        
        return np.array([self.mass, self.length])

    def set_origin(self, origin: np.ndarray|list):
        '''
            Sets pendulum origin (hang-point)
        
            Parameters:
            -------------------------
            origin: Pendulum hang-point
        '''

        if type(origin) is list:
            self.origin = np.array(origin, dtype=np.float32)
        else:
            self.origin = origin

class DoublePendulum():
    '''
        A double pendulum, consisting of two coupled pendulum objects.
    '''
    def __init__(self, pendulum1: Pendulum, pendulum2: Pendulum):
        
        self.pendulum1 = pendulum1
        self.pendulum2 = pendulum2

    def get_properties(self):
        '''Returns the properties of the double pendulum'''
        mass1, mass2 = self.pendulum1.mass, self.pendulum2.mass
        length1, length2 = self.pendulum1.length, self.pendulum2.length

        return np.array([mass1, mass2, length1, length2])
        
    def set_double_pendulum(self, theta1: float|int, w1: float|int
                            , theta2: float|int, w2: float|int):
        '''
            Sets upper pendulum, as well as origin for lower pendulum
            
            Parameters:
            ----------------
            theta1:  upper pendulum angle
            w1:      upper pendulum angular velocity
            theta2:  lower pendulum angle
            w2:      lower pendulum angular velocity
        '''
        # sets the state
        self.state = np.array([theta1, theta2, w1, w2], dtype=np.float32)
        # set upper pendulum
        self.origin = self.pendulum1.origin
        self.pendulum1.set_pendulum(theta=theta1, w=w1)
        # set lower pendulum origin
        xp, yp = self.pendulum1.get_cartesian_state()[:2]
        self.pendulum2.set_origin([xp, yp])
        self.pendulum2.set_pendulum(theta=theta2, w=w2)

    def get_cartesian_state(self):

        x1, y1, vx1, vy1 = self.pendulum1.get_cartesian_state() 

        x2, y2, vx2, vy2 = self.pendulum2.get_cartesian_state()
        vx2 = vx1 + vx2
        vy2 = vy1 + vy2

        return (x1,y1,x2,y2), (vx1,vy1,vx2,vy2)
    
if __name__ == '__main__':
    # instantiate the two pendula making up the double pendulum
    pendulum1 = Pendulum(mass=1, length=1, origin=[0,0])
    pendulum2 = Pendulum(mass=1, length=1)
    # instantiate the double pendulum
    double_pendulum = DoublePendulum(pendulum1=pendulum1, pendulum2=pendulum2)
    double_pendulum.set_double_pendulum(theta1=np.pi/4, w1=0
                                        , theta2=np.pi/6, w2=0)
    print("double pendulum instantiated")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    x_points = [double_pendulum.pendulum1.origin[0]
                , double_pendulum.pendulum1.x
                , double_pendulum.pendulum2.x]
    y_points = [double_pendulum.pendulum1.origin[1]
                , double_pendulum.pendulum1.y
                , double_pendulum.pendulum2.y]
    
    double_pendulum_line, = ax.plot(x_points, y_points, 'o-')

    plt.show()
    print("double pendulum plotted")