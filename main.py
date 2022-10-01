from datetime import datetime
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import math as math
import timeit as speed


def pendulum_one_step(X,t,m = 1 ,L = 1,b = 0.1,g = 9.81):
    X_1 = X[0]
    X_2 = X[1]
    X_1_dot = X_2
    X_2_dot = - g/L * math.sin(X_1) - b/m * X_2
    return X_1_dot,X_2_dot

def euler_pendulum_sim(theta_init, t, m = 1 ,L = 1,b = 0.1,g = 9.81):
    theta1 = [theta_init[0]]
    theta2 = [theta_init[1]]
    dt = t[1] - t[0]
    for i, t_ in enumerate(t[:-1]):
        next_theta1 = theta1[-1] + theta2[-1] * dt
        next_theta2 = theta2[-1] - (b/(m*L**2) * theta2[-1] - g/L *
            np.sin(next_theta1)) * dt
        theta1.append(next_theta1)
        theta2.append(next_theta2)
    return np.stack([theta1, theta2]).T

def main():
    # Input constants 
    m = 1 # mass (kg)
    L = 1 # length (m)
    b = 0 # damping value (kg/m^2-s)
    g = 9.81 # gravity (m/s^2)
    delta_t = 0.02 # time step size (seconds)
    t_max = 100 # max sim time (seconds)
    theta1_0 = np.pi/2 # initial angle (radians)
    theta2_0 = 0 # initial angular velocity (rad/s)
    theta_init = (theta1_0, theta2_0)
    # Get timesteps
    t = np.linspace(0, t_max, int(t_max/delta_t))
    theta_vals_int = integrate.odeint(pendulum_one_step, theta_init, t)
    fig, axs = plt.subplots(2)
    fig.suptitle("Pendulum velocity and angle")
    axs[0].set_title("Using Ode")
    axs[0].plot(theta_vals_int[:,0], '-b', label="position")
    axs[0].plot(theta_vals_int[:,1], '-r', label='velocty')
    axs[0].legend(loc="upper left")
    
    theta_vals_euler = euler_pendulum_sim(theta_init, t)
    mse_pos = np.power(    theta_vals_int[:,0] - theta_vals_euler[:,0], 2).mean()
    mse_vel = np.power(    theta_vals_int[:,1] - theta_vals_euler[:,1], 2).mean()
    print("MSE Position:\t{:.4f}".format(mse_pos))
    print("MSE Velocity:\t{:.4f}\n\n\n".format(mse_vel))
    axs[1].set_title("Using Euler")
    axs[1].plot(theta_vals_int[:,0], '-b', label="position")
    axs[1].plot(theta_vals_int[:,1], '-r', label='velocty')
    axs[1].legend(loc="upper left")
    plt.show()

if __name__ == "__main__": main()