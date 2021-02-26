#!/usr/bin/env python

import sys
import os
import logging
sys.path.insert(0,'simulations/starcraft/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)



#SCM imports
import structeral_causal_modeling as scm


"""
StarCraft II implementation is adaptod from - https://github.com/simonmeister/pysc2-rl-agents
"""

#starcraft II imports
import shutil
from functools import partial
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from rl.agents.a2c.runner import A2CRunner
from rl.agents.a2c.agent import A2CAgent
from rl.networks.fully_conv import FullyConv
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv
# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])


from helpers.config_simulations import get_config

import platform
import random
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf


import matplotlib
matplotlib.use('Agg')

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def _save_if_training(agent, summary_writer, config, ckpt_path):
  if config.train:
    agent.save(ckpt_path)
    summary_writer.flush()
    sys.stdout.flush()


def main():

    config, _ = get_config()
    simu_agent = config.simu_agent
    train_causal = config.train_scm

    """Starcraft II Init and Config"""
    if simu_agent == 'starcraft':
        config.train = not config.eval
        os.system('python -m tensorflow.tensorboard --logdir=' + config.logdir)
        #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        ckpt_path = os.path.join(config.save_dir, config.experiment_id)
        summary_type = 'train' if config.train else 'eval'
        summary_path = os.path.join(config.summary_dir, config.experiment_id, summary_type)

        if config.train and config.ow:
            shutil.rmtree(ckpt_path, ignore_errors=True)
            shutil.rmtree(summary_path, ignore_errors=True)

        size_px = (config.res, config.res)
        env_args = dict(
            map_name=config.map,
            players=[sc2_env.Agent(sc2_env.Race.terran), 
                    sc2_env.Bot(sc2_env.Race.protoss, 
                               sc2_env.Difficulty.very_easy)],
            step_mul=config.step_mul,
            game_steps_per_episode=0,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=size_px, minimap=size_px),
                use_unit_counts=True,
                raw_resolution=size_px,
                ),
            )

        vis_env_args = env_args.copy()
        vis_env_args['visualize'] = config.vis
        num_vis = min(config.envs, config.max_windows)
        env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
        num_no_vis = config.envs - num_vis
        if num_no_vis > 0:
            env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)

        envs = SubprocVecEnv(env_fns)
        starcraft_graph = tf.Graph()
        starcraft_sess = tf.Session(graph=starcraft_graph, config=tf.ConfigProto(log_device_placement=True))
        summary_writer = tf.summary.FileWriter(summary_path)
        network_data_format = 'NHWC' if config.nhwc else 'NCHW'

        starcraft_agent = A2CAgent(
                starsess=starcraft_sess,
                network_data_format=network_data_format,
                value_loss_weight=config.value_loss_weight,
                entropy_weight=config.entropy_weight,
                learning_rate=config.lr,
                max_to_keep=config.max_to_keep
                )

        runner = A2CRunner(
            envs=envs,
            agent=starcraft_agent,
            train=config.train,
            summary_writer=summary_writer,
            discount=config.discount,
            n_steps=config.steps_per_batch,
            minimap_size=size_px
            )

        static_shape_channels = runner.preproc.get_input_channels()
        starcraft_agent.build(static_shape_channels, resolution=config.res, scope='starcraft2', graph=starcraft_graph)

        if os.path.exists(ckpt_path):
            starcraft_agent.load(ckpt_path)
        else:
            starcraft_agent.init()
        runner.reset()    
    """End of Starcraft II Initialization"""

    

    """Start agent training"""

    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    
    
    with tf.Session(config=sess_config) as csess:
        csess.run(tf.global_variables_initializer())

        """Starcraft II agent training"""
        if simu_agent == 'starcraft':
            starcraft_replay_obs = []
            starcraft_replay_act = []
            i = 0
            game_step = 0
            try:
                while True:
                    write_summary = config.train and i % config.summary_iters == 0

                    if i > 0 and i % config.save_iters == 0:
                        _save_if_training(starcraft_agent, summary_writer, config, ckpt_path)

                    result = runner.run_batch(train_summary=write_summary)
                    game_step += 1
                    if write_summary:

                        agent_step, loss, summary, batch_replay = result

                        if train_causal:
                            batch_replay = list(batch_replay.values())
                            obs_set = batch_replay[2].tolist()
                            action_set = batch_replay[4].tolist() 
                            
                            if len(starcraft_replay_obs) < 1:
                                starcraft_replay_obs = obs_set
                                starcraft_replay_act = action_set
                            else:
                                starcraft_replay_obs.extend(obs_set)
                                starcraft_replay_act.extend(action_set)
                                                    
                            if len(starcraft_replay_obs) >= config.data_size:

                                """generate why and why not explanations for a given state index of the batch data (here 0) and save to file"""
                                scm.process_explanations(starcraft_replay_obs, starcraft_replay_act, config, 0, game_step)    
                                starcraft_replay_obs = []
                                starcraft_replay_act = []            

                        summary_writer.add_summary(summary, global_step=agent_step)
                        print('iter %d: loss = %f' % (agent_step, loss))

                    i += 1

                    if 0 <= config.iters <= i:
                        break

            except KeyboardInterrupt:
                pass

            _save_if_training(starcraft_agent, summary_writer, config, ckpt_path)

            envs.close()
            summary_writer.close()

            print('mean score: %f' % runner.get_mean_score())


if __name__ == '__main__':
    main()


