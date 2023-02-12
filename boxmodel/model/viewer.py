import sys

from .base import (BoxModel, AnalyticBoxModel)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import matplotlib.patches as patches

class BoxModelViewer():
    """
    Displays the time series result of the Box Model, either at a single time or over the entire time interval.

    Parameters
    ----------
    boxmodel : BoxModel
        The box model to be displayed
    """
    def __init__(self, boxmodel: BoxModel):
        self.model = boxmodel
        self.numerical_solution = boxmodel.numerical_solution
        self.dt = self.numerical_solution.dt
        self.fps = 1/self.numerical_solution.dt
    
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

        for frame in self.numerical_solution.frames:
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
        
        frame = self.numerical_solution.frames[0]
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
            frame = self.numerical_solution.frames[i]

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

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.numerical_solution.frames), interval=self.dt*1000, blit=True)

        return anim   

class AnalyticBoxModelViewer(BoxModelViewer):
    def __init__(self, boxmodel: AnalyticBoxModel):
        super().__init__(boxmodel)
        self.analytical_solution = boxmodel.analytical_solution

    def show_analytical_and_numerical(self):
        fig, ax = plt.subplots(2,1)
        ax[0].plot(self.analytical_solution.time, self.analytical_solution.width, label="Analytical", linestyle='dashed')
        ax[0].plot(self.numerical_solution.time, self.numerical_solution.width, label="Numerical")
        ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("$x_N(t)$")
        ax[0].legend(loc='lower right')
        ax[1].plot(self.analytical_solution.time, self.analytical_solution.height, label="Analytical", linestyle='dashed')
        ax[1].plot(self.numerical_solution.time, self.numerical_solution.height, label="Numerical")
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("$h_N(t)$")
        ax[1].legend(loc='lower right')
        plt.show()

    def show_error(self):
        fig, ax = plt.subplots()
        ax.plot(self.numerical_solution.time, np.abs(self.numerical_solution.width - self.analytical_solution.width))
        ax.set_xlabel("$t$")
        ax.set_ylabel("$\epsilon$")
        plt.show()

    def show_error_loglog(self):
        fig, ax = plt.subplots()
        ax.plot(self.numerical_solution.time, np.abs(self.numerical_solution.width - self.analytical_solution.width))
        ax.set_xlabel("$t$")
        ax.set_ylabel("$\epsilon$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()
