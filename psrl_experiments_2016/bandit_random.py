'''
Script to run tabular experiments in batch mode.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd
import argparse
import sys

from src import environment
from src import finite_tabular_agents

from src.feature_extractor import FeatureTrueState
from src.experiment import run_finite_tabular_experiment_lite

if __name__ == '__main__':
    '''
    Run a tabular experiment according to command line arguments
    '''

    # Take in command line flags
    parser = argparse.ArgumentParser(description='Run tabular RL experiment')
    parser.add_argument('--trials', help='number of trials to evaluate', default=int(1e3))
    parser.add_argument('--epLen', help='length of episode', type=int, default=10)
    parser.add_argument('--nEps', help='number of episodes', type=int, default=10)
    parser.add_argument('--alg', help='Agent constructor', type=str)
    args = parser.parse_args()

    trial_rews = []
    for trial in range(args.trials):
        # Make the environment
        env = environment.make_randomMDP(nState=10, nAction=5, epLen=args.epLen)

        # Make the feature extractor
        f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

        # Make the agent
        alg_dict = {'PSRL': finite_tabular_agents.PSRL,
                    'Random': finite_tabular_agents.Random,
                    'PSRLunif': finite_tabular_agents.PSRLunif,
                    'OptimisticPSRL': finite_tabular_agents.OptimisticPSRL,
                    'GaussianPSRL': finite_tabular_agents.GaussianPSRL,
                    'UCBVI': finite_tabular_agents.UCBVI,
                    'BEB': finite_tabular_agents.BEB,
                    'BOLT': finite_tabular_agents.BOLT,
                    'UCRL2': finite_tabular_agents.UCRL2,
                    'UCFH': finite_tabular_agents.UCFH,
                    'EpsilonGreedy': finite_tabular_agents.EpsilonGreedy}

        agent_constructor = alg_dict[args.alg]

        agent = agent_constructor(env.nState, env.nAction, env.epLen,
                                  scaling=1.)

        # Run the experiment
        total_rew = run_finite_tabular_experiment_lite(agent, env, f_ext, args.nEps, seed=np.random.randint(2**31-1))
        trial_rews.append(total_rew)
    print("Average Total Reward: ", np.mean(trial_rews))

