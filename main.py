# Parameter Estimation for Growth Curves including Substrate and Product
# by J. M. Müller / 09/2019 - 01/2024

import logging

from estimator import Estimator
from utils.logger import logger


logger.info("Welcome to the Growth and Product Estimator!\n\n")


if __name__ == '__main__':
    estimator = Estimator(config_path='./config.yml')
    estimator.run_estimation()
