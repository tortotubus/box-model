import sys

from .base import (BoxModel, MultipleBoxModel, AnalyticBoxModel, DepositSolution)

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.collections as clt

LINE_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
LABELS = ['wave', 'concentration', 'settling_speed', 'froude']

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
        self.deposit_solution = boxmodel.deposit_solution
        self.dt = self.numerical_solution.dt
        self.fps = 1/self.numerical_solution.dt
    
    def show_time_series(self):
        time = self.numerical_solution.frames[:,0]
        height = self.numerical_solution.frames[:,2]
        width = self.numerical_solution.frames[:,1]
        concentration = self.numerical_solution.frames[:,5]

        fig, ax = plt.subplots(2,2)

        #fig.set_title(self.model.plot_title)

        ax[0][0].plot(time, width, color='b')
        ax[0][0].set_xlabel('$t$')
        ax[0][0].set_ylabel('Width')
        ax[0][0].grid(True)

        ax[0][1].plot(time, height, color='b')
        ax[0][1].set_xlabel('$t$')
        ax[0][1].set_ylabel('Height')
        ax[0][1].grid(True)

        ax[1][0].plot(time, concentration, color='b')
        ax[1][0].set_xlabel('$t$')
        ax[1][0].set_ylabel('$c_b$')
        ax[1][0].grid(True)

        deposit_x, deposit_y = self.deposit_solution.deposit(np.max(time))

        ax[1][1].plot(deposit_x, deposit_y, color='b')
        ax[1][1].set_xlabel('$x$')
        ax[1][1].set_ylabel('Height')
        ax[1][1].grid(True)

        fig.tight_layout()

        plt.show()

    def save_animation(self, filename: str):
        anim = self.animation()
        writer = animation.FFMpegWriter(fps=self.fps,codec='h264')
        anim.save(filename, writer=writer)

    def animation(self):
        fig, ax = plt.subplots()

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('Blues', 1e4)
        concentration = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(concentration, ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel("$c_b$", rotation=90)

        plt.title(self.model.plot_title)
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
        time_label = ax.text(max_x*.1,max_y*.9,"")

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
        patch = patches.PathPatch(path, facecolor=cmap(frame[5]), lw=1, alpha=1)

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
            patch.set_facecolor(cmap(frame[5]))
            time_label.set_text("time={0:.{1}f}s".format(frame[0],2))

            return patch, time_label

        fig.tight_layout()
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.numerical_solution.frames), interval=self.dt*1000, blit=True)

        plt.show()

    def animation_time_shots(self, n: int):
        fig, ax = plt.subplots()

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('Blues', 1e4)
        concentration = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(concentration, ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel("$c_b$", rotation=90)

        plt.title(self.model.plot_title + " ($dt$: {0:.{1}f})".format(self.numerical_solution.time[n],2))
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
        
        patch_arr = []

        for j in range(int(np.floor(len(self.numerical_solution.time) / n))):
            frame = self.numerical_solution.frames[0 + j*n]
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

            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=cmap(frame[5]), alpha=0.75, lw=1)
            patch_arr.append(patch)


        ax.plot(self.numerical_solution.frames[:,1], self.numerical_solution.frames[:,2], color='b', linestyle='--')

        collection = clt.PatchCollection(patch_arr, match_original=True)

        ax.add_collection(collection)

        plt.show()

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

class MultipleBoxModelViewer():

    def __init__(self, boxmodel: MultipleBoxModel, label: str):
        self.model = boxmodel
        self.numerical_solution = boxmodel.numerical_solution
        self.deposit_solution = boxmodel.deposit_solution
        self.n_waves = boxmodel.n_waves
        self.dt = self.numerical_solution.dt
        self.fps = 1/self.numerical_solution.dt

        self.labels = [] 
        
        if label == LABELS[0]: # Waves
            for i in range(self.n_waves):
                self.labels.append("Wave {}".format(i+1))

        elif label == LABELS[1]: # Concentration
            for i in range(self.n_waves):
                self.labels.append("$c_b$: {:.2f}".format(self.numerical_solution.frames[0,i,5]))

        elif label == LABELS[2]: # Settling Speed
            for i in range(self.n_waves):
                self.labels.append("$u$: {:.3f}".format(self.deposit_solution.u[i]))

        elif label == LABELS[3]: # Froude
            for i in range(self.n_waves):
                self.labels.append("Fr: {:.2f}".format(self.model.froude[i]))
    
    def show_width(self):
        fig, ax = plt.subplots()
        
        time = self.numerical_solution.frames[:,0,0]

        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_N(t)$')
        ax.grid(True)

        for i in range (self.n_waves):
            width = self.numerical_solution.frames[:,i,1] - self.numerical_solution.frames[:,i,3]
            ax.plot(time, width, label=self.labels[i], linestyle='-', color=LINE_COLORS[i])

        fig.tight_layout()

        ax.legend(loc='right')
        
        plt.show()

    def show_height(self):

        fig, ax = plt.subplots()
        
        time = self.numerical_solution.frames[:,0,0]

        ax.set_xlabel('$t$')
        ax.set_ylabel('Height')
        ax.grid(True)

        for i in range (self.n_waves):
            height = self.numerical_solution.frames[:,i,2]
            ax.plot(time, height, label=self.labels[i], linestyle='-', color=LINE_COLORS[i])

        fig.tight_layout()

        ax.legend(loc='right')
        
        plt.show()

    def show_concentration(self):

        fig, ax = plt.subplots()
        
        time = self.numerical_solution.frames[:,0,0]

        ax.set_xlabel('$t$')
        ax.set_ylabel('$c_b$')
        ax.grid(True)      

        for i in range (self.n_waves):
            concentration = self.numerical_solution.frames[:,i,5]
            ax.plot(time, concentration, label=self.labels[i], linestyle='-', color=LINE_COLORS[i])

        fig.tight_layout()
        ax.legend(loc='right')
        plt.show()

    def show_deposit_total(self):

        fig, ax = plt.subplots()
        
        time = self.numerical_solution.frames[:,0,0]
        
        deposit_x, deposit_y = self.deposit_solution.deposit(np.max(time))

        ax.plot(deposit_x, deposit_y, color='b')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$h_s(t)$')
        ax.grid(True)

        fig.tight_layout()        
        plt.show()

    def show_deposit_stackplot(self):

        fig, ax = plt.subplots()
        
        time = self.numerical_solution.frames[:,0,0]
        
        deposit_x, deposit_y = self.deposit_solution.deposits(np.max(time))
        
        ax.stackplot(deposit_x, deposit_y, colors=LINE_COLORS, labels=self.labels)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$h_s(t)$')
        ax.grid(True)

        fig.tight_layout()

        ax.legend(loc='right')
        
        plt.show()

    def show_deposit_by_wave(self):

        fig, ax = plt.subplots()
        
        time = self.numerical_solution.frames[:,0,0]
        
        deposit_x, deposit_y = self.deposit_solution.deposits(np.max(time))

        for i in range(self.n_waves):
            ax.plot(deposit_x, deposit_y[i,:], color=LINE_COLORS[i], label=self.labels[i])

        ax.set_xlabel('$x$')
        ax.set_ylabel('$h_s(t)$')
        ax.grid(True)

        ax.legend(loc='right')

        fig.tight_layout()        
        plt.show()

    def show_time_series(self):

        fig, ax = plt.subplots(2,2)
        
        time = self.numerical_solution.frames[:,0,0]

        ax[0][0].set_xlabel('$t$')
        ax[0][0].set_ylabel('Width')
        ax[0][0].grid(True)

        ax[0][1].set_xlabel('$t$')
        ax[0][1].set_ylabel('Height')
        ax[0][1].grid(True)

        ax[1][0].set_xlabel('$t$')
        ax[1][0].set_ylabel('$c_b$')
        ax[1][0].grid(True)      
        
        deposit_x, deposit_y = self.deposit_solution.deposit(np.max(time))

        ax[1][1].plot(deposit_x, deposit_y, color='b')
        ax[1][1].set_xlabel('$x$')
        ax[1][1].set_ylabel('Height')
        ax[1][1].grid(True)

        

        for i in range (self.n_waves):
            height = self.numerical_solution.frames[:,i,2]
            width = self.numerical_solution.frames[:,i,1] - self.numerical_solution.frames[:,i,3]
            concentration = self.numerical_solution.frames[:,i,5]

            ax[0][0].plot(time, width, label=self.labels[i], linestyle='-', color=LINE_COLORS[i])
            ax[0][1].plot(time, height, label=self.labels[i], linestyle='-', color=LINE_COLORS[i])
            ax[1][0].plot(time, concentration, label=self.labels[i], linestyle='-', color=LINE_COLORS[i])

        fig.tight_layout()

        ax[0][0].legend(loc='right')
        ax[0][1].legend(loc='right')
        ax[1][0].legend(loc='right')
        
        plt.show()

    def save_animation(self, filename: str):
        anim = self.animation()
        writer = animation.FFMpegWriter(fps=self.fps,codec='h264')
        anim.save(filename, writer=writer)

    def animation(self):
        fig, ax = plt.subplots()
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('Blues', 1e4)
        concentration = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(concentration, ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel("$c_b$", rotation=90)

        plt.title(self.model.plot_title)
        plt.ylabel("$y$")
        plt.xlabel("$x$")

        min_x = sys.maxsize
        max_x = -sys.maxsize
        max_y = -sys.maxsize

        for i in range(self.n_waves):
            min_tail_x = np.min(self.numerical_solution.frames[:,i,3])
            max_head_x = np.max(self.numerical_solution.frames[:,i,1])
            max_head_y = np.max(self.numerical_solution.frames[:,i,2])

            if min_tail_x < min_x:
                min_x = min_tail_x
            
            if max_head_x > max_x:
                max_x = max_head_x

            if max_head_y > max_y:
                max_y = max_head_y

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(0, max_y)
        
        frame = self.numerical_solution.frames[0]
        time_label = ax.text((np.abs(max_x - min_x)*.1) + min_x, max_y*.9, "time={0:.{1}f}".format(frame[0,0],2))
        
        patch_arr = []

        for i in range(self.n_waves):
            verts = [
                (frame[i,3],frame[i,4]),
                (frame[i,1],frame[i,4]),
                (frame[i,1],frame[i,2]),
                (frame[i,3],frame[i,2]),
                (frame[i,3],frame[i,4]),
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=cmap(frame[i,5]), lw=1)
            patch_arr.append(patch)

        collection = clt.PatchCollection(patch_arr, match_original=True)

        ax.add_collection(collection)

        def animate(i):
            
            frame = self.numerical_solution.frames[i]
            patch_arr = []
            colors = []
            for j in range(self.n_waves):
                verts = [
                    (frame[j,3],frame[j,4]),
                    (frame[j,1],frame[j,4]),
                    (frame[j,1],frame[j,2]),
                    (frame[j,3],frame[j,2]),
                    (frame[j,3],frame[j,4]),
                ]

                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.CLOSEPOLY,
                ]
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor='r', lw=1)
                patch_arr.append(patch)
                colors.append(cmap(frame[j,5]))

            collection.set_paths(patch_arr)
            collection.set_facecolor(colors)
            time_label.set_text("time={0:.{1}f}s".format(frame[0,0],2))

        fig.tight_layout()
        anim = animation.FuncAnimation(fig, animate, frames=len(self.numerical_solution.frames), interval=self.dt*1000, blit=False)

        plt.show()

        #anim = self.animation()
        #writer = animation.FFMpegWriter(fps=self.fps,codec='h264')
        #anim.save("filename", writer=writer)
        #ret


    def animation_time_shots(self, n: int):
        fig, ax = plt.subplots()
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('Blues', 1e4)
        concentration = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(concentration, ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel("$c_b$", rotation=90)

        #plt.title(self.model.plot_title + " ($dt$: {0:.{1}f})".format(self.numerical_solution.time[n],2))
        plt.ylabel("$y$")
        plt.xlabel("$x$")

        min_x = sys.maxsize
        max_x = -sys.maxsize
        max_y = -sys.maxsize

        for i in range(self.n_waves):
            min_tail_x = np.min(self.numerical_solution.frames[:,i,3])
            max_head_x = np.max(self.numerical_solution.frames[:,i,1])
            max_head_y = np.max(self.numerical_solution.frames[:,i,2])

            if min_tail_x < min_x:
                min_x = min_tail_x
            
            if max_head_x > max_x:
                max_x = max_head_x

            if max_head_y > max_y:
                max_y = max_head_y

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(0, max_y)
        
        patch_arr = []

        for j in range(int(np.floor(len(self.numerical_solution.time) / n))):
            frame = self.numerical_solution.frames[0 + j*n]
            for i in range(self.n_waves):
                verts = [
                    (frame[i,3],frame[i,4]),
                    (frame[i,1],frame[i,4]),
                    (frame[i,1],frame[i,2]),
                    (frame[i,3],frame[i,2]),
                    (frame[i,3],frame[i,4]),
                ]

                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.CLOSEPOLY,
                ]

                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=cmap(frame[i,5]), alpha=0.75, lw=1)
                patch_arr.append(patch)

        for i in range(self.n_waves):
            ax.plot(self.numerical_solution.frames[:,i,1], self.numerical_solution.frames[:,i,2], color='b', linestyle='--')
            ax.plot(self.numerical_solution.frames[:,i,3], self.numerical_solution.frames[:,i,2], color='b', linestyle='--')

        collection = clt.PatchCollection(patch_arr, match_original=True)

        ax.add_collection(collection)
        fig.tight_layout()

        plt.show()
