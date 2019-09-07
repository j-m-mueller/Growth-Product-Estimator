from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt

# data:
t_data = [0, 2, 5, 8, 10]
x_data = [0.02, 0.5, 2, 6, 7]
s_data = [23, 22.5, 16, 8, 3]
p_data = [0, 2, 6, 20, 50]

# initial parameter estimates:
# initial concentration estimates:
x0, s0, p0 = 0.02, 20, 0

# initial parameter estimates:
mumax0 = 0.5
yxs0 = 0.5
ks0 = 0.05
px0 = 0.2
pmu0 = 0.4
params0 = [mumax0, yxs0, ks0, px0, pmu0]

# model:
def model(vals, t, *args):
    x = vals[0]
    s = vals[1]
    p = vals[2]

    mumax = args[0]
    ks = args[1]
    yxs = args[2]

    px = args[3]
    pmu = args[4]

    dxdt = x * mumax * s/(ks + s)
    dsdt = -dxdt * 1/yxs
    dpdt = x*px + pmu*dxdt

    return [dxdt, dsdt, dpdt]

# integration:
def estimator(params):
    params = tuple(params)
    # curr_x0 = params[0]
    # curr_s0 = params[1]
    course = odeint(model, [x_data[0], s_data[0], p_data[0]], t_data, args=params)
    x_sim = course[:, 0]
    s_sim = course[:, 1]
    p_sim = course[:, 2]
    deltas =[(np.square(xsim - xdat) + np.square(ssim - sdat) + np.square(psim - pdat)) for xsim, xdat, ssim, sdat, \
             psim, pdat in zip(x_sim, x_data, s_sim, s_data, p_sim, p_data)]
    delta_sum = np.sum(deltas)
    return delta_sum

# result = minimize(estimator, tuple(params0), method='SLSQP')
result = differential_evolution(estimator, bounds=[(0.01, 1), (0, 1), (0.01, 20), (0, 10000), (0, 10000)])
fin_params = result['x']

print("Optimization results: %s" % result)
print("Final parameters: %s" % fin_params)

# simulate complete course with fitted parameters:
t = np.linspace(0, 10, 100)
course = odeint(model, [x_data[0], s_data[0], p_data[0]], t, args=tuple(fin_params))

# plot creation:
plt.plot(t, course[:, 0], 'k:')
plt.plot(t, course[:, 1], 'r--')
plt.plot(t, course[:, 2], 'b-')
plt.plot(t_data, x_data, 'ko')
plt.plot(t_data, s_data, 'ro')
plt.plot(t_data, p_data, 'bo')
plt.xlabel('Time [h]')
plt.ylabel('X / S / P [g/L]')
plt.legend(['X', 'S', 'P', 'X_data', 'S_data', 'P_data'])
plt.show()
