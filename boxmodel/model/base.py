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
    froude : float
        Constant Froude number to be used
    time : float
        Time at which the simulation is to be started
    """
    def __init__(self, front: float, back: float, height: float, velocity: float, froude: float, time: float):
        self.front = front
        self.back = back
        self.width = np.abs(front-back)
        self.height = height
        self.velocity = velocity
        self.froude = froude
        self.time = time
    
    @property
    def numerical_solution(self):
        pass

    @property
    def deposit_solution(self):
        pass

    @property
    def plot_title(self):
        pass

    @abstractmethod
    def solve(self):
        pass

class MultipleBoxModel(ABC):
    """
    Base class for box models with multiple wave fronts
    
    Parameters
    ----------
    n_waves : int
        Number of waves to be considered
    front : float
        Position of the wave fronts
    back : float
        Position of the back of the wave
    height : float
        Starting height of the wave
    velocity : float
        Starting velocity of the wave
    froude : float
        Constant Froude number to be used
    time : float
        Time at which the simulation is to be started
    """
    def __init__(self, n_waves: int, front: float, back: float, height: float, velocity: float, froude: float, time: float):
        self.n_waves = n_waves
        self.front = front
        self.back = back
        self.width = np.abs(front-back)
        self.height = height
        self.velocity = velocity
        self.froude = froude
        self.time = time
    
    @property
    def numerical_solution(self):
        pass

    @property
    def deposit_solution(self):
        pass

    @property
    def n_waves(self):
        pass
    
    @property
    def plot_title(self):
        pass

    @abstractmethod
    def solve(self):
        pass

class MultipleBoxModelSolution():
    """
    Contains the time series of the solved model. This class is designed to be consumed by BoxModelViewer.

    Parameters
    ----------
    frames : int
        The n number of frames which will be contained
    dt : float
        The size of the time step between frames
    n_waves : int
        The number of waves to be considered

    Attributes
    ----------
    frames : array_like, shape (n,6) 
        The time series result of the solved model
    n_waves : int
        The number of waves to be considered
    dt : float
        This value is used for animations. It will be used to display frames at a rate of one nondimensional time to one second

    """
    def __init__(self, frames: int, dt: float, n_waves: int):
        self.frames = np.empty((frames, n_waves, 6), dtype = float)
        self.n_waves = n_waves
        self.dt = dt

    def frame(self, index: int, wave: int, time: float, head: Tuple[float, float], tail: Tuple[float, float], concentration: float):
        self.frames[index][wave][0] = time
        self.frames[index][wave][1] = head[0]
        self.frames[index][wave][2] = head[1]
        self.frames[index][wave][3] = tail[0]
        self.frames[index][wave][4] = tail[1]
        self.frames[index][wave][5] = concentration

    @property
    def frames(self):
        return self._frames
    
    @property
    def time(self):
        return self._frames[:,0,0]
    
    @frames.setter
    def frames(self, value):
        self._frames = value

class BoxModelSolution():
    """
    Contains the time series of the solved model. This class is designed to be consumed by BoxModelViewer.

    Parameters
    ----------
    frames : int
        The n number of frames which will be contained
    dt : float
        The size of the time step between frames

    Attributes
    ----------
    frames : array_like, shape (n,6) 
        The time series result of the solved model
    dt : float
        This value is used for animations. It will be used to display frames at a rate of one nondimensional time to one second

    """
    def __init__(self, frames: int, dt: float):
        self.frames = np.empty((frames, 6), dtype=float)
        self.dt = dt

    def frame(self, index: int, time: float, head: Tuple[float, float], tail: Tuple[float, float], concentration: float):
        self.frames[index][0] = time
        self.frames[index][1] = head[0]
        self.frames[index][2] = head[1] 
        self.frames[index][3] = tail[0]
        self.frames[index][4] = tail[1]
        self.frames[index][5] = concentration

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

class DepositSolution():
    """
    This class contains the recovered solution h_b for the deposit on the floor. It may also be consumed by various BoxModelViewer.

    Parameters
    ----------
    solution : BoxModelSolution
        The solution of the box model from which we recover the deposit height
    u : float
        The nondimensional settling speed of the particulate consider
    n : int
        The number of discrete locations in which we wish to recover sediment height
    start : float
        The first location in which we wish to recover sediment height
    end : float
        The final location in which we wish to recover sediment height
    """
    def __init__(self, solution: BoxModelSolution, u: float, n: int, start: float, end: float):
        self.solution = solution
        self.u = u
        self.n = n
        self.start = start
        self.end = end
    
    def deposit(self, time: float):
        deposit_height = np.zeros(self.n)
        deposit_x = np.linspace(self.start, self.end, self.n)

        def delta_b(x: float, i: int):
            if (self.solution.frames[i][3] <= x <= self.solution.frames[i][1]):
                return 1
            else:
                return 0

        for i in range(1, len(self.solution.frames[:,0])):
            if self.solution.frames[i][0] <= time:
                delta_t = self.solution.time[i] - self.solution.time[i-1]
                for j in range(len(deposit_x)):
                    deposit_height[j] += self.u * self.solution.frames[i][5] * delta_t * delta_b(deposit_x[j], i)

        return deposit_x, deposit_height
    
class MultipleDepositSolution():
    """
    This class contains the recovered solution h_b for the deposit on the floor. It may be consumed by MultipleBoxModelViewer.

    Parameters
    ----------
    solution : BoxModelSolution
        The solution of the box model from which we recover the deposit height
    u : float
        The nondimensional settling speed of the particulate consider
    n : int
        The number of discrete locations in which we wish to recover sediment height
    start : float
        The first location in which we wish to recover sediment height
    end : float
        The final location in which we wish to recover sediment height
    """
    def __init__(self, solution: MultipleBoxModelSolution, u: float, n: int, start: float, end: float):
        self.solution = solution
        self.u = u
        self.n = n
        self.start = start
        self.end = end

    def deposits(self, time: float):
        deposit_height = np.zeros((self.solution.n_waves, self.n))
        deposit_x = np.linspace(self.start, self.end, self.n)

        for n in range(self.solution.n_waves):

            def delta_b(x: float, i: int):
                if (self.solution.frames[i][n][3] <= x <= self.solution.frames[i][n][1]):
                    return 1
                else:
                    return 0

            for i in range(1, len(self.solution.frames[:,0])):
                if self.solution.frames[i][n][0] <= time:
                    delta_t = self.solution.time[i] - self.solution.time[i-1]
                    for j in range(len(deposit_x)):
                        deposit_height[n][j] += self.u[n] * self.solution.frames[i][n][5] * delta_t * delta_b(deposit_x[j], i)

        return deposit_x, deposit_height
          
    def deposit(self, time: float):
        deposit_height = np.zeros(self.n)
        deposit_x = np.linspace(self.start, self.end, self.n)

        for n in range(self.solution.n_waves):

            def delta_b(x: float, i: int):
                if (self.solution.frames[i][n][3] <= x <= self.solution.frames[i][n][1]):
                    return 1
                else:
                    return 0

            for i in range(1, len(self.solution.frames[:,0])):
                if self.solution.frames[i][n][0] <= time:
                    delta_t = self.solution.time[i] - self.solution.time[i-1]
                    for j in range(len(deposit_x)):
                        deposit_height[j] += self.u[n] * self.solution.frames[i][n][5] * delta_t * delta_b(deposit_x[j], i)

        return deposit_x, deposit_height
                
                

