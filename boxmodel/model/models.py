from .base import (
        BoxModel, MultipleBoxModel, 
        BoxModelSolution, MultipleBoxModelSolution,
        DepositSolution, MultipleDepositSolution
    )

from scipy.integrate import solve_ivp
import numpy as np

CONCENTRATION_THRESHOLD = 0.0005

class MultipleBoxModelWithConcentration(MultipleBoxModel):
    def __init__(self, front, back, height, velocity, concentration, u, froude, time):
        if not (len(front) == len(back) == len(height) == len(velocity) == len(concentration) == len(u)):
            raise ValueError
        
        for i in range(len(front)):
            if (front[i] <= back[i]):
                raise ValueError
            if (height[i] <= 0):
                raise ValueError
            if (concentration[i] < 0 or concentration[i] > 1):
                raise ValueError
            if (u[i] < 0):
                raise ValueError

        super().__init__(len(front), front, back, height, velocity, froude, time)

        self.concentration = concentration
        self.u = u
        self.plot_title = "Multiple Currents with Concentration"

    def solve(self, time: float, dt: float):

        def box_model(t: float, z: float):
            lhs = np.empty(self.n_waves * 3)

            for i in range(self.n_waves):
                x_n, x_t, c_b = z[0 + 3*i : 3 + 3*i]
                V = self.height[i] * self.width[i]
                u_s = self.u[i]
                fr = self.froude[i]

                lhs[0 + 3*i] =   fr * (np.sqrt(np.abs(V * c_b) / np.abs(x_n - x_t))) # dx_N(t)
                lhs[1 + 3*i] = - fr * (np.sqrt(np.abs(V * c_b) / np.abs(x_n - x_t))) # dx_T(t)
                lhs[2 + 3*i] = - (u_s * c_b * np.abs(x_n - x_t)) / V                 # dc_b(t)

            return lhs
        
        def concentration_threshold(t: float, z: float):
            cb_above_threshold = 0
            for i in range(self.n_waves):
                c_b = z[2 + 3*i]
                if c_b >= CONCENTRATION_THRESHOLD:
                    cb_above_threshold = 1

            return cb_above_threshold

        concentration_threshold.terminal = True 

        def current_interaction(t: float, z: float):
            for i in range(self.n_waves):
                xn_i, xt_i = z[0 + 3*i : 2 + 3*i]
                for j in range(self.n_waves):
                    if i != j:
                        xn_j, xt_j = z[0 + 3*j : 2 + 3*j]
                        if (xt_i <= xt_j <= xn_i) or (xt_i <= xn_j <= xn_i):
                            return 0
            return 1
        
        current_interaction.terminal = False

        initial_conditions = np.empty(self.n_waves * 3)

        for i in range(self.n_waves):
            initial_conditions[(0 + 3*i) : (3 + 3*i)] = self.front[i], self.back[i], self.concentration[i]

        sol = solve_ivp(
            box_model, 
            t_span=[self.time,time], 
            y0=initial_conditions, 
            #t_eval=np.arange(1, time, dt), 
            method='RK45',
            atol=np.inf,
            rtol=np.inf,
            max_step=dt,
            first_step=dt,
            events=[concentration_threshold,current_interaction]
        )

        self.numerical_solution = MultipleBoxModelSolution(frames=len(sol.y[0]), dt=dt, n_waves=self.n_waves)

        for i in range(len(sol.y[0])):
            t = sol.t[i]
            for j in range(self.n_waves):
                xN = sol.y[0 + 3*j][i]
                xT = sol.y[1 + 3*j][i]
                cN = sol.y[2 + 3*j][i]
                hN = (self.height[j] * self.width[j]) / np.abs(xN-xT)
                self.numerical_solution.frame(index=i, wave=j, time=t, head=(xN, hN), tail=(xT, 0.), concentration=cN)
        
        self.deposit_solution = MultipleDepositSolution(
            solution=self.numerical_solution, 
            u=self.u, 
            n=int(1e3), 
            start=np.min(self.numerical_solution.frames[-1,:,3]), 
            end=np.max(self.numerical_solution.frames[-1,:,1])
        )

    def numerical_solution(self):
        return self.numerical_solution
    
    def deposit_solution(self):
        return self.deposit_solution
    
    def plot_title(self):
        return self.plot_title
    
    def n_waves(self):
        return super().n_waves


class BoxModelWithConcentration(BoxModel):
    """
    """
    def __init__(self, front: float, back: float, height: float, velocity: float, time: float, concentration: float, u: float):
        super().__init__(front, back, height, velocity, time)
        self.concentration = concentration
        self.u = u
        self.plot_title = "Box Model Current with Single Concentration"

    def solve(self, time: float, dt: float):

        def box_model(t: float, z: float):
            froude = np.sqrt(2)
            volume = np.abs(self.height*self.width)

            x, c = z

            dx =  froude * np.sqrt(np.abs((volume*c)/x))
            dc = -(self.u * x * c) / volume

            return [dx, dc]

        #sol = solve_ivp(box_model, t_span=[self.time,time], y0=[self.front, self.concentration], t_eval=np.arange(1, time, dt))

        sol = solve_ivp(
            box_model, 
            t_span=[self.time,time], 
            y0=[self.front, self.concentration], 
            #t_eval=np.arange(1, time, dt), 
            method='RK45',
            atol=np.inf,
            rtol=np.inf,
            max_step=dt,
            first_step=dt,
        )

        self.numerical_solution = BoxModelSolution(frames=len(sol.y[0]), dt=dt)

        for i in range(len(sol.y[0])):
            t = sol.t[i]
            xN = sol.y[0][i]
            cN = sol.y[1][i]
            hN = (self.height*self.width)/sol.y[0][i]
            self.numerical_solution.frame(index=i, time=t, head=(xN,hN), tail=(0.,0.), concentration=cN)

        self.deposit_solution = DepositSolution(solution=self.numerical_solution, u=self.u, n=1000, start=0., end=sol.y[0][-1])

    def numerical_solution(self):
        return self.numerical_solution
    
    def deposit_solution(self):
        return self.deposit_solution

    def plot_title(self):
        return self.plot_title

    def analytical_solution(self):
        return self.analytical_solution