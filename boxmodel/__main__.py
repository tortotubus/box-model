from model.models import (BoxModelWithConcentration)
from model.viewer import (BoxModelViewer)

if __name__ == '__main__':

    model = BoxModelWithConcentration(
        front = 1.,
        back = 0.,
        height = 1.,
        velocity = 0.,
        time = 0.,
        concentration=0.01,
        u = 0.5)

    model.solve(time=30, dt=0.1)
    viewer = BoxModelViewer(model)
    viewer.show()    
