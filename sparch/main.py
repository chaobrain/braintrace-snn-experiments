"""
This is the script used to run experiments.
"""

import os
import sys
import time
# time.sleep(5 * 60 * 60)  # wait for 5 hours before starting

# Linux
sys.path.append('/mnt/d/codes/projects/brainscale')
sys.path.append('/mnt/d/codes/projects/brainstate')
sys.path.append('/mnt/d/codes/projects/brainevent')

# windows
sys.path.append('D:/codes/projects/brainscale')
sys.path.append('D:/codes/projects/brainstate')
sys.path.append('D:/codes/projects/brainevent')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from exp import Experiment
from args import add_training_options, add_model_options
from bst_utils import MyArgumentParser


def parse_args():
    parser = MyArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    return args


def main():
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python main.py -h` for details.
    """

    # Get experiment configuration from parser
    args = parse_args()

    # Instantiate class for the desired experiment
    experiment = Experiment(args)

    # Run experiment
    experiment.f_train()


if __name__ == "__main__":
    main()
