from . import model
from .numerics.integrate import _ivp 
import numpy as np
import matplotlib.pyplot as plt

def yprime(t: float, y: float):
    return y - np.power(t,2) + 1

def yexact(t: float): 
    return np.power(t + 1, 2) - 0.5*np.exp(t)

def main():

    print(np.finfo(float).eps)

    a = 0.
    b = 2.
    y0 = 0.5

    atol = 1e400

    n = 9

    error = np.empty(n)
    step_size = np.empty(n)

    for i in range(n):
        hmin = np.abs(a-b)/np.power(2,i+2)
        hmax = hmin
        solver = _ivp.RKF45(yprime, a, b, y0, atol, hmax, hmin)
        steps = int(np.abs(a - b) / hmin)

        for j in range(steps):
            solver.step()

        error[i] = np.abs(solver.get_y()-yexact(solver.get_t()))
        step_size[i] = hmin

    fig, ax = plt.subplots()

    ax.plot(step_size,error)
    ax.set_yscale('log')
    ax.set_ylabel('$|\epsilon|$')
    ax.set_xscale('log')
    ax.set_xlabel('$\Delta t$')

    error_log = np.log10(error)
    step_size_log = np.log10(step_size)
    fit = np.polynomial.polynomial.polyfit(step_size_log,error_log, 1)

    print(fit[1])
    ax.set_title("Order From Polyfit: {:.6f}".format(fit[1]))
    plt.show()
    
if __name__ == '__main__':
    main()
