from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# simulation parameters:
t = np.linspace(0, 10, 100)
mumax = 0.5
yxs = 0.5
ks = 0.05

# model:
def model(vals, t):
    x = vals[0]
    s = vals[1]
    dxdt = x * mumax * s/(ks + s)
    dsdt = -dxdt * 1/yxs
    return [dxdt, dsdt]

# integration:
course = odeint(model, [0.2, 20], t)

# plot creation:
plt.plot(t, course[:, 0], 'k:')
plt.plot(t, course[:, 1], 'r--')
plt.show()