'''
Run this script to train a model
'''

from setup.load_args import LoadArgs
from solver import Solver # use this for Kitti data

# Load arguments from config file, change subdirectory to parent directory of config.yaml
kwargs = LoadArgs()

# Instantiate training object which loads all model parameters
solver = Solver(**kwargs)

# Train model
solver.train()