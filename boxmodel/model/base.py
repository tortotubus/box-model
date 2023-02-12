from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

class BoxModel(ABC):
    """
    Base class for all BoxModels
    
    Parameters
    ----------
    front : float
        Position of the wave front
    back : float
        Position of the back of the wave volume
    height : float
        Starting height of the wave
    velocity : float
        Starting velocity of the wave
    time : float
        Time at which the simulation is to be started
    """
    def __init__(self, front: float, back: float, height: float, velocity: float, time: float):
        self.front = front
        self.back = back
        self.width = np.abs(front-back)
        self.height = height
        self.velocity = velocity
        self.time = time
    
    @property
    def numerical_solution(self):
        pass

    @abstractmethod
    def solve(self):
        pass

class AnalyticBoxModel(BoxModel):
    """
    Base class for all BoxModels
    
    Parameters
    ----------
    front : float
        Position of the wave front
    back : float
        Position of the back of the wave volume
    height : float
        Starting height of the wave
    velocity : float
        Starting velocity of the wave
    time : float
        Time at which the simulation is to be started
    """
    def __init__(self, front: float, back: float, height: float, velocity: float, time: float):
        super().__init__(front=front, back=back, height=height, velocity=velocity, time=time)
    
    @property
    def analytical_solution():
        pass

class BoxModelSolution():
    """
    Contains the time series of the solved model

    Parameters
    ----------
    frames : int
        The number of frames which will be contained
    dt : float
        The size of the time step between frames

    Attributes
    ----------
    frames : 
        The time series result of the solved model
    """
    def __init__(self, frames: int, dt: float):
        self.frames = np.empty((frames, 5), dtype=float)
        self.dt = dt

    def frame(self, index: int, time: float, head: Tuple[float, float], tail: Tuple[float, float]):
        self.frames[index][0] = time
        self.frames[index][1] = head[0]
        self.frames[index][2] = head[1] 
        self.frames[index][3] = tail[0]
        self.frames[index][4] = tail[1]

    @property
    def frames(self):
        return self._frames

    @property
    def time(self):
        return self._frames[:,0]
    
    @property
    def width(self):
        return np.abs(self._frames[:,0] - self._frames[:,3])

    @property
    def height(self):
        return np.abs(self._frames[:,2] - self._frames[:,4])

    @frames.setter
    def frames(self, value):
        self._frames = value

 