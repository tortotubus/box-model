from .base import (BoxModel, AnalyticBoxModel, BoxModelSolution)

from scipy.integrate import solve_ivp
#from boxmodel.numerics.integrate import solve_ivp
import numpy as np

class BoxModelWithConcentration(BoxModel):
    """
    """
    def __init__(self, front: float, back: float, height: float, velocity: float, time: float, concentration: float, u: float):
        super().__init__(front, back, height, velocity, time)
        self.concentration = concentration
        self.u = u

    def solve(self, time: float, dt: float):

        def box_model(t: float, z: float):
            froude = np.sqrt(2)
            volume = self.height*self.width

            x, c = z

            dx =  froude * np.sqrt((volume*c)/x)
            dc = -(self.u * x * c) / volume

            return [dx, dc]

        sol = solve_ivp(box_model, t_span=[self.time,time], y0=[self.front, self.concentration], atol=1e5)
        self.numerical_solution = BoxModelSolution(frames=len(sol.y[0]), dt=dt)

        for i in range(len(sol.y[0])):
            t = sol.t[i]
            xN = sol.y[0][i]
            #cN = sol.y[1][i]
            hN = (self.height*self.width)/sol.y[0][i]
            self.numerical_solution.frame(index=i, time=t, head=(xN,hN), tail=(0.,0.))

    def numerical_solution(self):
        return self.numerical_solution

    def analytical_solution(self):
        return self.analytical_solution

class BoxModelWithSource(AnalyticBoxModel):
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
        
        """ Solve the box model numerically, first."""

        def box_model(t: float, x: float):
            froude = np.sqrt(2)
            dxdt =  froude * np.sqrt((self.q*np.power(t,self.alpha))/x)
            return dxdt

        sol = solve_ivp(box_model, t_span=[self.time,time], y0=[self.front], max_step=dt, min_step=dt, atol=1e5)
        self.numerical_solution = BoxModelSolution(frames=len(sol.y[0]), dt=dt)

        for i in range(len(sol.y[0])):
            t = sol.t[i]
            xN = sol.y[0][i]
            hN = (self.q*np.power(t,self.alpha))/xN
            self.numerical_solution.frame(index=i, time=t, head=(xN,hN), tail=(0.,0.))

        """ Next, 'solve' the box model analytically, using a known solution for x_N(t) """

        def exact_xN(t: float) -> float:
            return np.power(np.sqrt(2) * (3/2) * np.sqrt(self.q) * (np.power(t, 0.5 * self.alpha + 1) / (self.alpha / 2 + 1)) + np.power(self.front, 1.5), 2/3)

        self.analytical_solution = BoxModelSolution(frames=len(sol.y[0]), dt=dt)

        for i in range(len(sol.y[0])):
            t = i*dt + self.time
            xN = exact_xN(t)
            hN = (self.q*np.power(t,self.alpha))/xN
            self.analytical_solution.frame(index=i, time=t, head=(xN,hN), tail=(0.,0.))
            

    def numerical_solution(self):
        return self.numerical_solution

    def analytical_solution(self):
        return self.analytical_solution