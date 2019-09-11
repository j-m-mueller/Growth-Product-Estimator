from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
import sys

from functions import model, error_calculator, plot_results, data_df
from config import mode, simulated_courses_file

print("\nWelcome to GrowthCalculator!\n")

# initial parameter estimates:
# initial concentration estimates:
X0, S0, P0 = 0.02, 20, 0

# initial parameter estimates:
mumax0 = 0.5
YXS0 = 0.5
KS0 = 0.05
px0 = 0.2
pmu0 = 0.4
params0 = [mumax0, YXS0, KS0, px0, pmu0]

# parameter bounds:
parameter_bounds = [(0.01, 1), (0, 1), (0.01, 20), (0, 10000), (0, 10000)]

if __name__ == '__main__':
    # error minimization methods:
    if mode == 'opt_min':
        result = minimize(estimator, tuple(params0), method='SLSQP')
    elif mode == 'diff_eq':
        result = differential_evolution(error_calculator, bounds=parameter_bounds, disp=True)
    else:
        print("Please choose a valid optimization method (diff_eq or opt_min)!")
        sys.exit()

    # isolate optimized parameters:
    fin_params = result['x']

    # print results to console:
    print("\nOptimization results:\n%s" % result)
    print("\nFinal parameters:")
    print("µmax: %s, Y(X/S): %s, KS: %s, pX: %s, pmu: %s." % tuple(['{:.3f}'.format(param) for param in fin_params]))

    # simulate complete course with fitted parameters:
    t = np.linspace(0, 10, 100)
    course = odeint(model, [data_df['X'].iloc[0], data_df['S'].iloc[0], data_df['P'].iloc[0]/2, data_df['P'].iloc[0]/2,
                            data_df['P'].iloc[0]], t, args=tuple(fin_params))

    # create results DataFrame:
    sim_df = pd.DataFrame(data=np.column_stack((t, course[:, 0], course[:, 1], course[:, 2], course[:, 3],
                                                course[:, 4])), columns=['t', 'X', 'S', 'P(X)', 'P(mu)', 'P'])

    # calculate proportions of biomass- and growth-related product formation, respectively:
    prop_px_to_p = sim_df['P(X)'].iloc[-1]/sim_df['P'].iloc[-1] * 100
    prop_pmu_to_p = sim_df['P(mu)'].iloc[-1]/sim_df['P'].iloc[-1] * 100

    # print results:
    print("\nFinal proportion of biomass-related product formation [P(X)]: %s%%, proportion of growth-related product"
          " formation [P(mu)]: %s%%." % ('{:.2f}'.format(prop_px_to_p), '{:.2f}'.format(prop_pmu_to_p)))

    # save optimized courses of X, S, and P:
    sim_df.to_csv(simulated_courses_file, sep='\t')

    # visualize optimization results in a plot:
    plot_results(data_df, sim_df)
