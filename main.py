import numpy as np

from boxmodel.model.models import MultipleBoxModelWithConcentration
from boxmodel.model.viewer import MultipleBoxModelViewer

import matplotlib.pyplot as plt

centers = np.array([-3, 3])
width = np.array([1, 1])
height = np.array([1.,0.7])

noses = centers + 0.5*width
tails = centers - 0.5*width

bm = MultipleBoxModelWithConcentration(
    time = 0.0,
    front = noses,
    back = tails,
    height = [1.,.7],
    velocity = [0,0],
    concentration = [1., 1.,],
    u = [0.02, 0.02],
    froude = [np.sqrt(2), np.sqrt(2)]
)

bm.solve(time=20., dt=0.05)

viewer = MultipleBoxModelViewer(bm, "label")
viewer.animation()
viewer.show_deposit_total()
viewer.show_deposit_by_wave()