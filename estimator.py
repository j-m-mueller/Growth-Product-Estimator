"""Collective methods for modeling."""

import logging
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from typing import List

from utils.config_model import ParameterConfig
from utils.logger import logger


class Estimator:
    """Class to perform parameter estimation."""

    def __init__(self,
                 config_path: str = './config.yml'):
        """Initialize the class and read the configuration."""
        self._config = self._parse_config(config_path=config_path)
        
        self._input_data = pd.DataFrame()
        self._simulated_data = pd.DataFrame()

        self._optimization_result: dict = {}
        self._parameters: list = []

    def run_estimation(self):
        """Wrapper method to execute the estimation and plot the output."""
        self._read_data()
        self._estimate_parameters()
        self._simulate_course()

        if self._config.input_output.plot_output:
            self._plot_results()
    
    def _parse_config(self,
                      config_path: str) -> ParameterConfig:
        """Parses the provided Parameter Configuration.

        :param config_path: path to the configuration.
        :return: ParameterConfig object.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file.read())

        return ParameterConfig(**config)

    def _read_data(self) -> pd.DataFrame:
        """Reads input data."""
        self._input_data = pd.read_csv(self._config.input_output.input_path, sep='\t')

    def _estimate_parameters(self):
        """Initialize the parameter estimation based on the provided configuration."""
        match self._config.model.estimation_mode:
            case 'opt_min':
                logger.info("Estimating parameters via SciPy's optimize.minimize...\n")
                results = minimize(
                    self._get_least_squares, 
                    tuple(self._config.model.parameters.initial_estimates), 
                    method='SLSQP'
                )
            case 'diff_evol':
                logger.info("Estimating parameters via Differential Evolution...\n")
                results = differential_evolution(
                    self._get_least_squares, 
                    bounds=self._config.model.parameters.bounds, 
                    disp=True
                )
            case _:
                logger.error("Please choose a valid optimization method (diff_eq or opt_min)!")
                sys.exit()

        self._optimization_result = results
        self._parameters = results['x']
            
        # print results to console:
        logger.info(f"Optimization results:\n\n{results}\n")
        logger.info("Final parameters:\n\nµmax: %s, Y(X/S): %s, KS: %s, pX: %s, pµ: %s.\n" 
                    % tuple(['{:.3f}'.format(param) for param in self._parameters]))
    
    def _simulate_course(self) -> None:
        """Simulate the courses of the variables for the estimated parameter set."""
        # simulate complete course with fitted parameters
        t = np.linspace(0, 10, 100)
        
        course = odeint(self._model_values, 
                        [
                            self._input_data["X"].iloc[0], 
                             self._input_data["S"].iloc[0], 
                             self._input_data["P"].iloc[0]/2,
                             self._input_data["P"].iloc[0]/2,
                             self._input_data["P"].iloc[0]
                        ],
                        t,
                        args=tuple(self._parameters))
        
        course_df = pd.DataFrame(course, columns=['X', 'S', 'P(X)', 'P(mu)', 'P'])
        
        # create results DataFrame
        sim_df = pd.DataFrame(t, columns=["t"])
        sim_df = pd.concat([sim_df, course_df], axis=1)
    
        # calculate proportions of biomass- and growth-related product formation, respectively
        prop_px_to_p = sim_df['P(X)'].iloc[-1] / sim_df['P'].iloc[-1]
        prop_pmu_to_p = sim_df['P(mu)'].iloc[-1] / sim_df['P'].iloc[-1]
    
        # log results
        logger.info(f"Product proportions:\n\n"
                    f"- final proportion of biomass-related product formation [secondary product, P(X)]: {prop_px_to_p:.2%}\n"
                    f"- proportion of growth-related product formation [primary product, P(µ)]: {prop_pmu_to_p:.2%}\n")
        
        # save optimized courses of X, S, and P
        if self._config.input_output.save_output:
            sim_df.to_csv(self._config.input_output.output_path, sep='\t')
            logger.info(f"Stored simulated output at {self._config.input_output.output_path}\n")

        self._simulated_data = sim_df

    @staticmethod
    def _model_values(vals: List[float],
                      t: float,
                      *args: List[float]) -> List[float]:
        """
        Calculate the current differentials.

        :param vals: current model values.
        :param t: time.
        :param *args: model parameters.

        :return: list of differentials (rates).
        """ 
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

    def _get_least_squares(self, params: dict) -> float:
        """
        Calculates the least square error for a given dataset.
    
        :param params: initial parameter estimates.
        :return: least square error.
        """   
        # gather model result and isolate simulated values (use initial values as estimates)
        course = odeint(self._model_values, 
                        [
                            self._input_data["X"].iloc[0], 
                             self._input_data["S"].iloc[0], 
                             self._input_data["P"].iloc[0]/2,
                             self._input_data["P"].iloc[0]/2,
                             self._input_data["P"].iloc[0]
                        ],
                        self._input_data["t"].tolist(), 
                        args=tuple(params))
        
        # gather least square error
        deltas = [(np.square(xsim - xdat) + np.square(ssim - sdat) + np.square(psim - pdat)) 
                  for xsim, xdat, ssim, sdat, psim, pdat 
                  in zip(
                      course[:, 0],  # simulated X
                      self._input_data["X"].tolist(), 
                      course[:, 1],  # simulated S
                      self._input_data["S"].tolist(), 
                      course[:, 4],  # simulated P
                      self._input_data["P"].tolist()
                  )]
        
        return np.sqrt(np.sum(deltas))

    def _plot_results(self) -> None:
        """Generate a Seaborn Plot for comparison of modelled and input data."""
        sns.set()

        # plot traces
        for col, symbol_sim, symbol_input in zip(
            ['X', 'S', 'P'],
            ['k:', 'r--', 'b-'], 
            ['ko', 'ro', 'bo']
        ):
            plt.plot(self._simulated_data['t'], self._simulated_data[col], symbol_sim)
            plt.plot(self._input_data['t'], self._input_data[col], symbol_input)

        # highlight plot areas according to product terms
        plt.fill_between(self._simulated_data['t'], 
                         self._simulated_data['P'], 
                         self._simulated_data['P(mu)'], 
                         facecolor='skyblue')
        plt.fill_between(self._simulated_data['t'], 
                         self._simulated_data['P(mu)'], 
                         0, 
                         facecolor='steelblue')
    
        plt.suptitle('Growth Product Parameter Estimation')
        plt.title('Proportions of P(X) and P(µ) plotted in light and dark blue, respectively.')
        plt.xlabel('Time [h]')
        plt.ylabel('X / S [g/L] and P [mg/L]')
        plt.legend(['X (sim.)', 'S (sim.)', 'P (sim.)', 'X (data)', 'S (data)', 'P (data)'])
        plt.show()
