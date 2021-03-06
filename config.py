# config file:

# mode of parameter estimation:
mode = 'diff_evol'  # choose from 'diff_evol' (differential evolution) or 'opt_min' (optimize.minimize)
data_file = './data/demo-data.csv'  # specify file name to be used for parameter estimation
simulated_courses_file = './output/simulated-data.csv'  # file name to be used for simulated data

# initial parameter estimates:
mumax0 = 0.5  # maximal specific growth rate [e.g. in 1/h]
YXS0 = 0.5  # yield coefficient of biomass per substrate [dimensionless]
KS0 = 0.05  # Monod constant [e.g. in g/L]
px0 = 0.2  # term for biomass-related product formation (secondary product)
pmu0 = 0.4  # term for growth-related product formation (primary product)
params0 = [mumax0, YXS0, KS0, px0, pmu0]

# parameter bounds [parameters in the order of the params0 list; each term specifies minimal and maximal value]:
parameter_bounds = [(0.01, 1), (0, 1), (0.01, 20), (0, 10000), (0, 10000)]