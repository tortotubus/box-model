from boxmodel import boxmodel as bm

def main():
    model = bm.BoxModelWithSource(front=1., back=0., height=1., velocity=0., time=0., alpha=0)
    model.solve(10., 0.05)
    viewer = bm.BoxModelViewer(model)
    viewer.show()
if __name__ == "__main__":
    main()