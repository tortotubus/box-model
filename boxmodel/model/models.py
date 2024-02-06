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
        
        current_interaction.terminal = True

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

        # If collision occurs, find the index/time where it happens
        if len(sol.t_events[1]) != 0:

            # Identify time of collision
            terminal_t = sol.t_events[1][0]
            arg_t = np.argwhere(sol.t == terminal_t)[0][0] - 1
            print(f"Collision occurs at t={terminal_t}, index={arg_t}")

            # Find the rows for fronts and concentrations
            arg_fronts = np.array([[i,i-1] for i in range(1,self.n_waves*3,3)]).flatten()
            arg_c = [i for i in range(2,self.n_waves*3,3)]

            # Put fronts and concentrations in their own matrices
            fronts = sol.y[arg_fronts,:arg_t]
            concentrations = sol.y[arg_c,:arg_t]

            # Create a new volume matrix
            volumes = np.zeros_like(concentrations)

            for i in range(self.n_waves):
                volumes[i,:] = (self.front[i] - self.back[i])*self.height[i]

            # Find the indicies where we want to create new boxes
            arg_insert_fronts = np.array([i for i in range(2,len(fronts)-1,2)])
            arg_insert_concentrations = np.array([i for i in range(1,len(concentrations))])

            # Create special front pairs for solving in modified box model
            collision_fronts_pairs = []

            # Insert new box fronts
            for i in arg_insert_fronts:
                right = fronts[i,:]
                left = fronts[i-1,:]
                middle = np.mean([right,left],axis=0)

                collision_fronts_pairs.append([left,middle,right])

                fronts = np.insert(
                    arr=fronts,
                    obj=i,
                    values=np.vstack((left,middle,middle,right)),
                    axis=0,
                )

                self.n_waves += 2

            # Insert new concentrations and volumes
            for i in arg_insert_concentrations:
                concentrations = np.insert(
                    arr=concentrations,
                    obj=i,
                    values=np.array([concentrations[i-1,:], concentrations[i,:]]),
                    axis=0,
                )

                volumes = np.insert(
                    arr=volumes,
                    obj=i,
                    values=[[1e-12],[1e-12]],
                    #values=[[1e-12],[1e-12]],
                    axis=0,
                )

            #print(fronts)
            #print(concentrations)
            #print(volumes)

            def box_model_collision(t: float, z: float):
                left, center, right = z[0:3]
                c1,c2,c3,c4 = z[3:7]
                V1,V2,V3,V4 = z[7:11]
                u_s = 0

                left_ = 0.45682*V1 + 0.06707*V2 + -0.53149*V3 + -0.46849*V4 + -0.17260*c3 + -0.05621*c4
                center_ = 0.51032*V1 + 0.00415*V2 + -0.54307*V4 + -0.05879*c1 + 0.11532*c2 + -0.06338*c3 + 0.02701*c4
                right_ = 0.48520*V1 + 0.10959*V2 + 0.36019*V3 + -0.53834*V4 + -0.18608*c1 + 0.33033*c2 + -0.08190*c3 + 0.20897*c4

                V1_ = -0.07008*V1 +  0.08357*V2 +  0.02249*V3 + -0.09118*V4 +  0.26945*V1*u_s +  1.83365*V2*u_s +  0.01149*V3*u_s + -0.49722*V4*u_s
                V2_ =  0.07016*V1 + -0.07842*V2 + -0.02846*V3 +  0.09190*V4 + -0.23123*V1*u_s + -2.26293*V2*u_s +  0.43293*V3*u_s +  0.44054*V4*u_s
                V3_ =  0.06196*V1 + -0.13175*V2 +  0.03626*V3 +  0.08926*V4 + -0.20196*V1*u_s + -1.55399*V2*u_s + -0.17158*V3*u_s +  0.40881*V4*u_s
                V4_ = -0.06134*V1 +  0.13179*V2 + -0.03585*V3 + -0.09032*V4 +  0.15986*V1*u_s +  1.96789*V2*u_s + -0.26530*V3*u_s + -0.34043*V4*u_s

                return np.array([left_,center_,right_,0,0,0,0,V1_,V2_,V3_,V4_])

            left,center,right = collision_fronts_pairs[0]
            print(left[-1],center[-1],right[-1])

            collision_solution = solve_ivp(
                box_model_collision, 
                t_span=[sol.t[arg_t],time], 
                y0=np.hstack((left[-1],center[-1],right[-1],concentrations[:,-1],volumes[:,-1])), 
                method='RK45',
                atol=np.inf,
                rtol=np.inf,
                max_step=dt,
                first_step=dt,
            )

            import matplotlib.pyplot as plt

            print(collision_solution.y.T)
            plt.plot(collision_solution.t, collision_solution.y.T[:,:])
            plt.show()

            y = np.vstack((fronts[0],fronts[1],concentrations[0],volumes[0]))
            # Recombine into old format
            for i in range(1,self.n_waves):
                y = np.vstack((y,fronts[0 + i*2],fronts[1 + i*2], concentrations[i], volumes[i]))
            
            t = sol.t[:arg_t]

            self.numerical_solution = MultipleBoxModelSolution(frames=len(y[0]), dt=dt, n_waves=self.n_waves)

            for i in range(len(y[0])):
                for j in range(self.n_waves):
                    xN = y[0 + 4*j][i]
                    xT = y[1 + 4*j][i]
                    cN = y[2 + 4*j][i]
                    vN = y[3 + 4*j][i]
                    hN = vN / np.abs(xN - xT)
                    self.numerical_solution.frame(index=i, wave=j, time=t[i], head=(xN, hN), tail=(xT, 0.), concentration=cN)


        # Collision did not occur
        else:
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