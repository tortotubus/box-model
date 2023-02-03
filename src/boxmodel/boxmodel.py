import sys

from abc import ABC, abstractmethod
from typing import Tuple

from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import matplotlib.patches as patches


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
    def solution(self):
        pass

    @abstractmethod
    def solve(self):
        pass

class BoxModelWave():
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

    @frames.setter
    def frames(self, value):
        self._frames = value

class BoxModelWithSource(BoxModel):
    """
    Implementation of the BoxModel with a source term. Rate of inflow is given by alpha.

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
    alpha : float
        The rate of inflow
    """
    def __init__(self, front: float, back: float, height: float, velocity: float, time: float, alpha: float):
        super().__init__(front, back, height, velocity, time)
        self.q = abs(front-back)*height
        self.alpha = alpha
    
    def solve(self, time: float, dt: float):
        
        def box_model(t: float, x: float):
            froude = np.sqrt(2)
            dxdt =  froude * np.sqrt((self.q*np.power(t,self.alpha))/x)
            return dxdt

        sol = solve_ivp(lambda t, x: box_model(t, x), t_span=[self.time,time], y0=[self.front], t_eval=np.arange(self.time, time, dt))
        print(sol)
        self.frames = BoxModelWave(frames=len(sol.y[0]), dt=dt)

        for i in range(len(sol.y[0])):

            t = sol.t[i]
            xN = sol.y[0][i]
            hN = (self.q*np.power(t,self.alpha))/xN

            self.frames.frame(index=i, time=t, head=(xN,hN), tail=(0.,0.))

    def solution(self):
        return self.frames

class BoxModelViewer():
    """
    Displays the time series result of the Box Model, either at a single time or over the entire time interval.

    Parameters
    ----------
    boxmodel : BoxModel
        The box model to be displayed
    """
    def __init__(self, boxmodel: BoxModel):
        self.frames = boxmodel.frames
        self.dt = self.frames.dt
        self.fps = 1/self.frames.dt
        self.model = boxmodel

    
    def save(self, filename: str):
        anim = self.animation()
        writer = animation.FFMpegWriter(fps=self.fps,codec='h264')
        anim.save(filename, writer=writer)

    def show(self):
        anim = self.animation()
        plt.show()

    def animation(self):
        fig, ax = plt.subplots()
        plt.title("Box Model")
        plt.ylabel("$h_N(t)$")
        plt.xlabel("$x_N(t)$")

        min_x = sys.maxsize
        min_y = sys.maxsize

        max_x = -sys.maxsize
        max_y = -sys.maxsize

        for frame in self.frames.frames:
            tail_x = frame[3]
            tail_y = frame[4]
            head_x = frame[1]
            head_y = frame[2]

            if tail_x < min_x:
                min_x = tail_x
            
            if tail_y < min_y:
                min_y = tail_y

            if head_x > max_x:
                max_x = head_x

            if head_y > max_y:
                max_y = head_y

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        frame = self.frames.frames[0]
        time_label = ax.text(0.5,0.85,"")

        verts = [
            # Tail
            (frame[3],frame[4]),
            (frame[1],frame[4]),
            (frame[1],frame[2]),
            (frame[3],frame[2]),
            (frame[3],frame[4]),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='blue', lw=1)

        def init():
            ax.add_patch(patch)
            return patch,

        def animate(i):
            frame = self.frames.frames[i]

            verts = [
                (frame[3],frame[4]),
                (frame[1],frame[4]),
                (frame[1],frame[2]),
                (frame[3],frame[2]),
                (frame[3],frame[4]),
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            patch.set_path(Path(verts, codes))
            time_label.set_text("time={0:.{1}f}s".format(frame[0],2))

            return patch,time_label

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.frames.frames), interval=self.frames.dt*1000, blit=True)

        return anim    