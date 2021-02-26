#-*- coding: utf-8 -*-
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


#simulations starcraft II
starcraft_arg = add_argument_group('Starcraft II Config')
starcraft_arg.add_argument('--experiment_id', type=str, default='1',
                    help='identifier to store experiment results')
starcraft_arg.add_argument('--logdir', type=str, default="/",
                    help='logdir')
starcraft_arg.add_argument('--eval', action='store_true',
                    help='if false, episode scores are evaluated')
starcraft_arg.add_argument('--ow', action='store_true',
                    help='overwrite existing experiments (if --train=True)')
starcraft_arg.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
starcraft_arg.add_argument('--vis', action='store_true',
                    help='render with pygame')
starcraft_arg.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open')
starcraft_arg.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')
starcraft_arg.add_argument('--envs', type=int, default=32,
                    help='number of environments simulated in parallel')
starcraft_arg.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')
starcraft_arg.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')
starcraft_arg.add_argument('--discount', type=float, default=0.95,
                    help='discount for future rewards')
starcraft_arg.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')
starcraft_arg.add_argument('--starcraftseed', type=int, default=123,
                    help='random seed')
starcraft_arg.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
starcraft_arg.add_argument('--nhwc', action='store_true',
                    help='train fullyConv in NCHW mode')
starcraft_arg.add_argument('--summary_iters', type=int, default=10,
                    help='record training summary after this many iterations')
starcraft_arg.add_argument('--save_iters', type=int, default=1000,
                    help='store checkpoint after this many iterations')
starcraft_arg.add_argument('--max_to_keep', type=int, default=5,
                    help='maximum number of checkpoints to keep before discarding older ones')
starcraft_arg.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy loss')
starcraft_arg.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')
starcraft_arg.add_argument('--lr', type=float, default=7e-4,
                    help='initial learning rate')
starcraft_arg.add_argument('--data_size', type=int, default=16,
                    help='batch_size for scm')   
starcraft_arg.add_argument('--save_dir', type=str, default=os.path.join('out','models'),
                    help='root directory for checkpoint storage')
starcraft_arg.add_argument('--summary_dir', type=str, default=os.path.join('out','summary'),
                    help='root directory for summary storage')
starcraft_arg.add_argument('--train_scm', type=bool, default=True,
                    help='train structeral equations')
starcraft_arg.add_argument('--simu_agent', type=str, default='starcraft',
                    help='agent to simulate')
starcraft_arg.add_argument('--scm_mode', type=str, default='train',
                    help='whether to train or infer scm')
starcraft_arg.add_argument('--scm_regressor', type=str, default='lr',
                    help='regressor types for training the scm, lr == linear regressor, dt = decision tree based regressors, mlp = neural based regressors')                                        


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


