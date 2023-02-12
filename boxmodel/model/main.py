import boxmodel as bm
import numpy as np
import numerics as num

def main():
    #alpha = 0.1
    #model = bm.BoxModelWithSource(front=1, back=0., height=1, velocity=0., time=1., alpha=alpha)
    #model.solve(time=10, dt=0.05)
    #viewer = bm.AnalyticBoxModelViewer(model)
    #viewer.show()
    #viewer.show_error_loglog()    

    def f(t: float, y:float):
        return 1

    #num.solve_ivp(f, 0)
    solver = num.OdeSolver(f, 0, 0, 0, 0, 0, 0)
    
if __name__ == "__main__":
    main()