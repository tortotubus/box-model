from abc import ABC, abstractmethod

import scipy.integrate

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import matplotlib.patches as patches

class BoxModel():
    def __init__(self, front: float = 1.0, back: float = 0, h0: float = 1.0, u0: float = 1.0, t: list[float, float, float] = [0,1,0.1]):
        self.front = front
        self.back = back
        self.x0 = abs(front-back)
        self.h0 = h0
        self.u0 = u0
        self.time_range = t
        self.timeseries = np.array([[0,0,1,1], [0,0,2,1], [0,0,3,1], [0,0,4,1], [0,0,5,1], [0,0,6,1],])
    
    def get_frame(self, index: int):
        return self.timeseries[index]
    
    def box_model(t: float, x: float):
        return t*x

    def solve():
        pass

class BoxModelViewer(ABC):
    def __init__(self, boxmodel: BoxModel):
        self.bm = boxmodel
    
    def show(self):
        fig, ax = plt.subplots()

        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

        for frame in self.bm.timeseries:
            if frame[0] < min_x:
                min_x = frame[0]
            
            if frame[1] < min_y:
                min_y = frame[1]

            if frame[2] > max_x:
                max_x = frame[2]

            if frame[3] > max_y:
                max_y = frame[3]


        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        frame = self.bm.timeseries[0]

        verts = [
            (frame[0],frame[1]),
            (frame[2],frame[1]),
            (frame[2],frame[3]),
            (frame[0],frame[3]),
            (frame[0],frame[1]),
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
            frame = self.bm.timeseries[i]
        
            verts = [
                (frame[0],frame[1]),
                (frame[2],frame[1]),
                (frame[2],frame[3]),
                (frame[0],frame[3]),
                (frame[0],frame[1]),
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

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.bm.timeseries), interval=500, blit=True)

        plt.show()

    def show_at_frame(self, index: int):
        fig, ax = plt.subplots()        
        frame = self.bm.timeseries[index]
        
        ax.set_xlim(frame[0],frame[2])
        ax.set_ylim(frame[1],2*frame[3])

        verts = [
            (frame[0],frame[1]),
            (frame[2],frame[1]),
            (frame[2],frame[3]),
            (frame[0],frame[3]),
            (frame[0],frame[1]),
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
        ax.add_patch(patch)
        plt.show()
    

class BoxModelNetCDF(ABC):
    @property
    def createNetCDF():
        pass

    @property
    def createBoxModel():
        pass

