from scipy.integrate import odeint
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# data:
t_data = [0, 2, 5, 8, 10]
x_data = [0.02, 0.5, 2, 6, 7]
s_data = [23, 22.5, 16, 8, 3]

# initial parameter estimates:
mumax0 = 0.5
yxs0 = 0.5
ks0 = 0.05
params0 = [mumax0, yxs0, ks0]

# model:
def model(vals, t, *args):
    x = vals[0]
    s = vals[1]

    mumax = args[0]
    ks = args[1]
    yxs = args[2]

    dxdt = x * mumax * s/(ks + s)
    dsdt = -dxdt * 1/yxs
    return [dxdt, dsdt]

# integration:
def estimator(params):
    params = tuple(params)
    course = odeint(model, [0.2, 20], t_data, args=params)
    x_sim = course[:, 0]
    s_sim = course[:, 1]
    deltas = np.sum([(np.square(xsim - xdat) + np.square(ssim - sdat)) for xsim, xdat, ssim, sdat in zip(x_sim, x_data,
                                                                                                   s_sim, s_data)])
    return deltas

fin_params = minimize(estimator, tuple(params0), method='SLSQP')

print("Optimization results: %s" % fin_params)
print("Final parameters: %s" % fin_params['x'])

# simulate complete course with fitted parameters:
t = np.linspace(0, 10, 100)
course = odeint(model, [0.2, 20], t, args=tuple(fin_params['x']))

# plot creation:
plt.plot(t, course[:, 0], 'k:')
plt.plot(t, course[:, 1], 'r--')
plt.plot(t_data, x_data, 'bo')
plt.plot(t_data, s_data, 'go')
plt.show()
