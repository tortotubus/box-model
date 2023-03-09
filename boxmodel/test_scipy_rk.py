from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def yprime(t: float, y: float):
    return y - np.power(t,2) + 1

def yexact(t: float): 
    return np.power(t + 1, 2) - 0.5*np.exp(t)

def test_rfk45():
    
    t_final = 2
    samples = 8
    arr = np.zeros(shape=(2,samples))

    for i in range(samples):
        step_size = np.power(2.,-i)
        final_index = int(t_final/step_size)

        sol = solve_ivp(
            fun=yprime, 
            t_span=[0,t_final],
            y0=[0.5],
            method='RK45',
            atol=np.inf,
            rtol=np.inf,
            max_step=step_size,
            first_step=step_size
        )

        arr[0,i] = step_size
        arr[1,i] = np.abs(yexact(t_final) - sol.y[0,final_index])

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')

    logarr = np.empty_like(arr)

    logarr[0,:] = np.log(arr[0,:])
    logarr[1,:] = np.log(arr[1,:])
    
    fit = np.polyfit(logarr[0,:], logarr[1,:], deg=1)
    print(fit)
    
    ax.set_ylabel('$|\epsilon|$')
    ax.set_xlabel('$h$')

    ax.set_title("Dormand-Prince Method Order: {}".format(fit[0]))
    ax.plot(arr[0,:], arr[1,:])
    plt.show()

if __name__ == "__main__":
    test_rfk45()