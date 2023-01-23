from box_model import BoxModel as bm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    model = bm.BoxModel()
    viewer = bm.BoxModelViewer(model)
    viewer.show()

if __name__ == "__main__":
    main()