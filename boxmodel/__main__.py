from model.models import (BoxModelWithConcentration, MultipleBoxModelWithConcentration)
from model.viewer import (BoxModelViewer, MultipleBoxModelViewer)

import numpy as np
import matplotlib as plt

if __name__ == '__main__':

#    plt.rcParams['lines.linewidth'] = 2
#    plt.rcParams['font.size'] = 18
#    plt.rcParams['axes.linewidth'] = 2
#    plt.rcParams['xtick.labelsize'] = 16
#    plt.rcParams['ytick.labelsize'] = 16
#    plt.rcParams['legend.fontsize'] = 10
#    plt.rcParams['savefig.format'] = 'pdf'

    n = 2

    demo = MultipleBoxModelWithConcentration(
        back          = np.array([-3.  ,2]),
        front         = np.array([-2.  ,3]),
        height        = np.array([ 1.  ,2.]),
        velocity      = np.zeros(n),
        concentration = np.array([ 1.  ,1.]),
        u             = np.array([ 0.05,0.15]),
        froude        = np.ones(n)*np.sqrt(2),
        time          = 0
    )

    demo.solve(time=50, dt=0.05)
#
#    #plt.rcParams["figure.figsize"] = (6.5,4)
#    #plt.rcParams["figure.figsize"] = (6.5,3)
#
    multi_viewer = MultipleBoxModelViewer(demo, 'waves')
#    #multi_viewer.animation_time_shots(10)
#    multi_viewer.animation()
    multi_viewer.save_animation("movie2.mp4")
#
#    n = 5
#
#    varied_settling_speed = MultipleBoxModelWithConcentration(
#        back          = np.ones(n) * -0.5,
#        front         = np.ones(n) *  0.5,
#        height        = np.ones(n),
#        velocity      = np.zeros(n),
#        concentration = np.ones(n),      
#        u             = np.linspace(.005,0.2,num=n),
#        froude        = np.ones(n) * np.sqrt(2),
#        time          = 0
#    )
#
#    varied_settling_speed.solve(time=100, dt=0.05)
#
#    #plt.rcParams["figure.figsize"] = (3.25,4)
#    plt.rcParams["figure.figsize"] = (6.5,2.25)
#
#    print(varied_settling_speed.numerical_solution.frames[-1,0,0])
#
#    multi_viewer = MultipleBoxModelViewer(varied_settling_speed, 'settling_speed')
#    multi_viewer.show_width()
#    #multi_viewer.show_height()
#    multi_viewer.show_deposit_by_wave()
#    #multi_viewer.show_deposit_stackplot()
#
#
#    n = 4
#
#    varied_froude = MultipleBoxModelWithConcentration(
#        back          = np.ones(n) * -0.5,
#        front         = np.ones(n) *  0.5,
#        height        = np.ones(n),
#        velocity      = np.zeros(n),
#        concentration = np.ones(n),      
#        u             = np.ones(n) * 0.1,
#        froude        = np.linspace(1.19,2,n),
#        time          = 0
#    )
#
#    #plt.rcParams["figure.figsize"] = (6.5,4)
#
#    varied_froude.solve(time=100, dt=0.05)
#
#    multi_viewer = MultipleBoxModelViewer(varied_froude, 'froude')
#
#    multi_viewer.show_width()
#    #multi_viewer.show_height()
#    #multi_viewer.show_time_series()
#    #multi_viewer.show_deposit_by_wave()
#    #multi_viewer.show_deposit_stackplot()
