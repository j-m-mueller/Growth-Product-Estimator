"""Logger setup."""

import logging
import sys


# set up logger and define log level
logger = logging.getLogger('GrowthProductLogger')
logger.setLevel(logging.INFO)

# create a stream handler that logs to stdout, and add it to the logger
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# format logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# add the handler to the logger
logger.addHandler(stream_handler)
