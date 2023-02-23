from scipy.integrate import solve_ivp
#from boxmodel.numerics.integrate import _ivp
#from boxmodel.numerics.integrate import solve_ivp
import numpy as np

def test_numerics_vectorized():
    def lotkavolterra(t, z):
        a, b, c, d = 1.5, 1, 3, 1
        x, y = z
        return [a*x - b*x*y, -c*y + d*x*y]

    sol = solve_ivp(lotkavolterra, [0,0.1], [5,10])

    print(sol)

if __name__ == "__main__":
    test_numerics_vectorized()