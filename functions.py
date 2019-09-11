from scipy.integrate import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import data_file

# read data from .csv file:
data_df = pd.read_csv(data_file, sep='\t')

# hypothetical experimental data:
t_data = data_df['t'].tolist()
X_data = data_df['X'].tolist()
S_data = data_df['S'].tolist()
P_data = data_df['P'].tolist()

# growth and production model:
def model(vals, t, *args):
    # isolate current concentrations:
    X = vals[0]
    S = vals[1]
    PX = vals[2]
    Pmu = vals[3]
    P = vals[4]

    # isolate parameters from arguments:
    mumax = args[0]
    KS = args[1]
    YXS = args[2]
    pX = args[3]
    pmu = args[4]

    # gather current differentials:
    dXdt = X * mumax * S/(KS + S)  # biomass growth
    dSdt = -dXdt * 1/YXS  # substrate consumption
    dPXdt = X*pX
    dPmudt = pmu*dXdt
    dPdt = dPXdt + dPmudt  # product formation (primary and secondary term)

    return [dXdt, dSdt, dPXdt, dPmudt, dPdt]

# calculation of model error:
def least_square_calculator(params):
    # convert parameters to tuple:
    params = tuple(params)

    # gather model result and isolate simulated values:
    course = odeint(model, [X_data[0], S_data[0], P_data[0]/2, P_data[0]/2, P_data[0]], t_data, args=params)
    X_sim = course[:, 0]
    S_sim = course[:, 1]
    PX_sim = course[:, 2]
    Pmu_sim = course[:, 3]
    P_sim = course[:, 4]

    # calculate error square sum:
    deltas = [(np.square(xsim - xdat) + np.square(ssim - sdat) + np.square(psim - pdat)) for xsim, xdat, ssim, sdat,
               psim, pdat in zip(X_sim, X_data, S_sim, S_data, P_sim, P_data)]
    delta_sum = np.sqrt(np.sum(deltas))

    return delta_sum

# plot creation:
def plot_results(data_df, sim_df):
    sns.set()
    plt.plot(sim_df['t'], sim_df['X'], 'k:')
    plt.plot(sim_df['t'], sim_df['S'], 'r--')
    plt.plot(sim_df['t'], sim_df['P'], 'b-')
    plt.plot(data_df['t'], data_df['X'], 'ko')
    plt.plot(data_df['t'], data_df['S'], 'ro')
    plt.plot(data_df['t'], data_df['P'], 'bo')

    plt.fill_between(sim_df['t'], sim_df['P'], sim_df['P(mu)'], facecolor='skyblue')
    plt.fill_between(sim_df['t'], sim_df['P(mu)'], 0, facecolor='steelblue')

    plt.suptitle('Result of Parameter Estimation:')
    plt.title('[proportions of P(X) and P(mu) plotted in light and dark blue, respectively]')
    plt.xlabel('Time [h]')
    plt.ylabel('X / S [g/L] and P [mg/L]')
    plt.legend(['X (sim.)', 'S (sim.)', 'P (sim.)', 'X (data)', 'S (data)', 'P (data)'])
    plt.show()
