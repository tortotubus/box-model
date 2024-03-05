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

        has_collided = np.full(shape=(self.n_waves-1),fill_value=False)
        qpc = 4 # Quantities per current
        arg_begin_extra = self.n_waves*qpc
        self.u += self.u
        self.froude += self.froude

        """
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
        """

        def box_model(t: float, z: float):

            lhs_base = np.zeros(self.n_waves*qpc)
            lhs_extra = np.zeros((self.n_waves - 1)*2*qpc)

            for i in range(self.n_waves-1):
                if has_collided[i]:
                    # Left Current
                    u_s_1 = self.u[i]
                    fr_1 = self.froude[i]
                    x_n_1, x_t_1, c1, V1 = z[0+qpc*i:qpc+qpc*i]

                    # Center Left Current
                    x_n_2, x_t_2, c2, V2 = z[arg_begin_extra+qpc*(i*2):arg_begin_extra+qpc*(i*2)+4]

                    # Center Right Current
                    x_n_3, x_t_3, c3, V3 = z[arg_begin_extra+qpc*(i*2+1):arg_begin_extra+qpc*(i*2+1)+4]

                    # Right Current
                    u_s_4 = self.u[(i+1)]
                    fr_4 = self.froude[(i+1)]
                    x_n_4, x_t_4, c4, V4 = z[0+qpc*(i+1):qpc+qpc*(i+1)]


                    # Calculate colliding system
                    left_outer_  = -1.08790*V1 + 0.52031*V2 + -1.53037*V3 +  0.24832*V4 +  0.51994*c1 + -0.89129*c2 +  0.64735*c3 + -0.50969*c4
                    left_        = 0.45682*V1 + 0.06707*V2 + -0.53149*V3 + -0.46849*V4 + -0.17260*c3 + -0.05621*c4
                    center_      = 0.51032*V1 + 0.00415*V2 + -0.54307*V4 + -0.05879*c1 +  0.11532*c2 + -0.06338*c3 +  0.02701*c4
                    right_       = 0.48520*V1 + 0.10959*V2 +  0.36019*V3 + -0.53834*V4 + -0.18608*c1 +  0.33033*c2 + -0.08190*c3 +  0.20897*c4
                    right_outer_ = 0.29398*V1 + 0.23460*V2 +  0.77733*V3 +  0.63647*V4 +  0.07809*c1 + -0.11973*c2 +  0.21491*c3

                    V1_ = -0.07008*V1 +  0.08357*V2 +  0.02249*V3 + -0.09118*V4 +  0.26945*V1*u_s_1 +  1.83365*V2*u_s_1 +  0.01149*V3*u_s_1 + -0.49722*V4*u_s_1
                    V2_ =  0.07016*V1 + -0.07842*V2 + -0.02846*V3 +  0.09190*V4 + -0.23123*V1*u_s_1 + -2.26293*V2*u_s_1 +  0.43293*V3*u_s_1 +  0.44054*V4*u_s_1
                    V3_ =  0.06196*V1 + -0.13175*V2 +  0.03626*V3 +  0.08926*V4 + -0.20196*V1*u_s_4 + -1.55399*V2*u_s_4 + -0.17158*V3*u_s_4 +  0.40881*V4*u_s_4
                    V4_ = -0.06134*V1 +  0.13179*V2 + -0.03585*V3 + -0.09032*V4 +  0.15986*V1*u_s_4 +  1.96789*V2*u_s_4 + -0.26530*V3*u_s_4 + -0.34043*V4*u_s_4

                    c1_ =  0.02317*c1 + -0.02383*c2 +  0.00000*c3 + 0.00000*c4 + -0.98547*(c1*u_s_1*np.abs(x_n_1-x_t_1))/V1 +  0.12777*(c2*u_s_1*np.abs(x_n_2-x_t_2))/V2 + 0.03985*(c3*u_s_4*np.abs(x_n_3-x_t_3))/V3 + -0.01183*(c4*u_s_4*np.abs(x_n_4-x_t_4))/V4 
                    c2_ =  0.47762*c1 + -0.48307*c2 + -0.26500*c3 + 0.25673*c4 +  0.14509*(c1*u_s_1*np.abs(x_n_1-x_t_1))/V1 +  0.37209*(c2*u_s_1*np.abs(x_n_2-x_t_2))/V2 + 4.99356*(c3*u_s_4*np.abs(x_n_3-x_t_3))/V3 + -0.86917*(c4*u_s_4*np.abs(x_n_4-x_t_4))/V4 
                    c3_ =  0.32447*c1 + -0.33807*c2 + -0.40245*c3 + 0.40252*c4 +  0.00000*(c1*u_s_1*np.abs(x_n_1-x_t_1))/V1 +  1.34207*(c2*u_s_1*np.abs(x_n_2-x_t_2))/V2 + 4.07762*(c3*u_s_4*np.abs(x_n_3-x_t_3))/V3 + -0.75077*(c4*u_s_4*np.abs(x_n_4-x_t_4))/V4 
                    c4_ = -0.01246*c1 +  0.01200*c2 + -0.03116*c3 + 0.03093*c4 + -0.00075*(c1*u_s_1*np.abs(x_n_1-x_t_1))/V1 + -0.08640*(c2*u_s_1*np.abs(x_n_2-x_t_2))/V2 + 0.22259*(c3*u_s_4*np.abs(x_n_3-x_t_3))/V3 + -0.99420*(c4*u_s_4*np.abs(x_n_4-x_t_4))/V4 

                    # Left Current                    
                    lhs_base[0+qpc*i] = left_ # dx_N(t)                    
                    lhs_base[1+qpc*i] = left_outer_ # dx_T(t)                    
                    lhs_base[2+qpc*i] = c1_ # dc_b(t)                
                    lhs_base[3+qpc*i] = V1_ # dV_b(t)

                    # Center Left Current
                    lhs_extra[0+qpc*(i*2)] = center_ # dx_N(t)                    
                    lhs_extra[1+qpc*(i*2)] = left_ # dx_T(t)                    
                    lhs_extra[2+qpc*(i*2)] = c2_ # dc_b(t)                
                    lhs_extra[3+qpc*(i*2)] = V2_ # dV_b(t)

                    # Center Right Current
                    lhs_extra[0+qpc*(i*2+1)] = right_ # dx_N(t)                    
                    lhs_extra[1+qpc*(i*2+1)] = center_ # dx_T(t)                    
                    lhs_extra[2+qpc*(i*2+1)] = c3_ # dc_b(t)                
                    lhs_extra[3+qpc*(i*2+1)] = V3_ # dV_b(t)

                    # Right Current
                    lhs_base[0+qpc*(i+1)] = right_outer_ # dx_N(t)
                    lhs_base[1+qpc*(i+1)] = right_ # dx_T(t)
                    lhs_base[2+qpc*(i+1)] = c4_ # dc_b(t)
                    lhs_base[3+qpc*(i+1)] = V4_ # dV_b(t)

                else:
                    # Left Current
                    u_s_l = self.u[i]
                    fr_l = self.froude[i]
                    x_n_l, x_t_l, c_b_l, V_l = z[0+qpc*i:qpc+qpc*i]

                    lhs_base[0+qpc*i] =   fr_l * (np.sqrt(np.abs(V_l * c_b_l) / np.abs(x_n_l - x_t_l))) # dx_N(t)
                    lhs_base[1+qpc*i] = - fr_l * (np.sqrt(np.abs(V_l * c_b_l) / np.abs(x_n_l - x_t_l))) # dx_T(t)
                    lhs_base[2+qpc*i] = - (u_s_l * c_b_l * np.abs(x_n_l - x_t_l)) / V_l                 # dc_b(t)
                    #lhs_base[3+qpc*i] = 0                                                               # dV_b(t)

                    # Right Current
                    u_s_r = self.u[(i+1)]
                    fr_r = self.froude[(i+1)]
                    x_n_r, x_t_r, c_b_r, V_r = z[0+qpc*(i+1):qpc+qpc*(i+1)]

                    lhs_base[0+qpc*(i+1)] =   fr_r * (np.sqrt(np.abs(V_r * c_b_r) / np.abs(x_n_r - x_t_r))) # dx_N(t)                    
                    lhs_base[1+qpc*(i+1)] = - fr_r * (np.sqrt(np.abs(V_r * c_b_r) / np.abs(x_n_r - x_t_r))) # dx_T(t)
                    lhs_base[2+qpc*(i+1)] = - (u_s_r * c_b_r * np.abs(x_n_r - x_t_r)) / V_r                 # dc_b(t)
                    #lhs_base[3+qpc*(i+1)] = 0                                                               # dV_b(t)


            return np.concatenate((lhs_base,lhs_extra))
            #return lhs_base


        # Terminate condition if concentration reaches below a certain threshold
        def concentration_threshold(t: float, z: float):
            cb_above_threshold = 0
            for i in range(self.n_waves):
                c_b = z[2 + qpc*i]
                if c_b >= CONCENTRATION_THRESHOLD:
                    cb_above_threshold = 1

            return cb_above_threshold

        concentration_threshold.terminal = True 

        # If currents have collided, stop IVP solver 
        def tracked_current_interaction(t: float, z: float):
            for i in range(self.n_waves-1):
                left_front, right_front = z[i*qpc], z[1+(i+1)*qpc]
                if left_front > right_front:
                    has_collided[i] = True
                    return 0
            return 1
        
        tracked_current_interaction.terminal = True

        # Set initial conditions vector
        ics_base = np.empty(self.n_waves*qpc)
        extra_box_fill = 0.01
        ics_extra = np.full(shape=(self.n_waves-1)*qpc*2, fill_value=extra_box_fill)

        for i in range(self.n_waves):
            ics_base[(0+qpc*i):(qpc+qpc*i)] = self.front[i], self.back[i], self.concentration[i], self.height[i]*np.abs(self.front[i]-self.back[i])

        ics = np.concatenate((ics_base, ics_extra))

        # Solve the system
        sol = solve_ivp(
            box_model,
            t_span=[self.time,time], 
            y0=ics, 
            method='RK45',
            atol=np.inf,
            rtol=np.inf,
            max_step=dt,
            first_step=dt,
            events=[tracked_current_interaction]
        )

        # If the system terminates prematurely, solve the system again from just before termination 
        t_final = sol.t[-1]

        # Capture solved system from before termination
        y = sol.y[:,:-2]
        t = sol.t[:-2]

        # Track how many new boxes are added
        new_boxes = 0

        while t_final < time:
            # Create new initial conditions from earlier solved system
            ics = y[:,-1]

            for i in range(len(has_collided)):
                if has_collided[i]:
                    new_boxes = new_boxes + 2

                    # Left Current
                    u_s_l = self.u[i]
                    fr_l = self.froude[i]
                    x_n_l, x_t_l, c_b_l, V_l = ics[0+qpc*i:qpc+qpc*i]
                    h_l = V_l / np.abs(x_n_l - x_t_l)

                    # Right Current
                    u_s_r = self.u[(i+1)]
                    fr_r = self.froude[(i+1)]
                    x_n_r, x_t_r, c_b_r, V_r = ics[0+qpc*(i+1):qpc+qpc*(i+1)]
                    h_r = V_r / np.abs(x_n_r - x_t_r)

                    center = np.mean([x_n_l,x_t_r])

                    # Center Left Current
                    ics[arg_begin_extra+0+qpc*(i*2)] = center                        # x_N(t_c)                    
                    ics[arg_begin_extra+1+qpc*(i*2)] = x_n_l                         # x_T(t_c)                    
                    ics[arg_begin_extra+2+qpc*(i*2)] = c_b_l                         # c_b(t_c)                
                    ics[arg_begin_extra+3+qpc*(i*2)] = h_l * np.abs(x_n_l - center)  # V_b(t_c)

                    # Center Right Current
                    ics[arg_begin_extra+0+qpc*(i*2+1)] = x_t_r                        # x_N(t_c)                    
                    ics[arg_begin_extra+1+qpc*(i*2+1)] = center                       # x_T(t_c)                    
                    ics[arg_begin_extra+2+qpc*(i*2+1)] = c_b_r                        # c_b(t_c)                
                    ics[arg_begin_extra+3+qpc*(i*2+1)] = h_r * np.abs(x_t_r - center) # V_b(t_c)

            # Solve system
            sol = solve_ivp(
                box_model,
                t_span=[t_final,time], 
                y0=ics, 
                method='RK45',
                atol=np.inf,
                rtol=np.inf,
                max_step=dt,
                first_step=dt,
                events=[tracked_current_interaction]
            )

            t_final = sol.t[-1]
            y = np.concatenate((y[:,:-2],sol.y[:,:-2]),axis=1)
            t = np.concatenate((t[:-2],sol.t[:-2]))

            #y = np.concatenate((y,sol.y[:,:-2]),axis=1)
            #t = np.concatenate((t,sol.t[:-2]))
            
        self.numerical_solution = MultipleBoxModelSolution(frames=len(t), dt=dt, n_waves=(self.n_waves + (self.n_waves-1)*2))

        for i in range(len(t)):
            for j in range(self.n_waves + (self.n_waves-1)*2):
                xN = y[0 + qpc*j][i]
                xT = y[1 + qpc*j][i]
                cN = y[2 + qpc*j][i]
                vN = y[3 + qpc*j][i]
                hN = vN / np.abs(xN-xT) if vN != extra_box_fill else 0
                self.numerical_solution.frame(index=i, wave=j, time=t[i], head=(xN, hN), tail=(xT, 0.), concentration=cN)
        
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