from model.models import (BoxModelWithConcentration)
from model.viewer import (BoxModelViewer)

if __name__ == '__main__':

    model = BoxModelWithConcentration(
        front = 1.,
        back = 0.,
        height = 1.,
        velocity = 0.,
        time = 0.,
        concentration=1,
        u = 0.05)

    model.solve(time=50, dt=0.1)
    viewer = BoxModelViewer(model)
    viewer.animation()
    #viewer.show()    
