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
        print(sol.y)

        self.frames = BoxModelWave(frames=len(sol.y[0]), dt=dt)

        for i in range(len(sol.y[0])):

            t = sol.t[i]
            xN = sol.y[0][i]
            hN = (self.q*np.power(t,self.alpha))/xN

            self.frames.frame(index=i, time=t, head=(xN,hN), tail=(0.,0.))

    def solution(self):
        return self.frames

class BoxModelViewer():
    def __init__(self, boxmodel: BoxModel):
        self.frames = boxmodel.frames
    
    def show(self):
        fig, ax = plt.subplots()

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
            return patch,
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.frames.frames), interval=self.frames.dt*1000, blit=True)

        plt.show()    