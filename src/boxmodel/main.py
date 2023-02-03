import boxmodel as bm
import numpy as np

def main():
    alpha = 0.1
    model = bm.BoxModelWithSource(front=1, back=0., height=1, velocity=0., time=1., alpha=alpha)
    model.solve(time=10, dt=0.05)
    viewer = bm.BoxModelViewer(model)
    viewer.show()    
    
if __name__ == "__main__":
    main()