#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from environments.obstacle_car.environment_vec import Environment_Vec


def train(params, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    with sess:

        def make_env():
            #env = gym.make(env_id)
            env = Environment_Vec(params, polar_coords=True)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env

        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        set_global_seeds(seed)
        policy = MlpPolicy
        #model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        #                   lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        #                   ent_coef=0.0,
        #                   lr=3e-4,
        #                   cliprange=0.2,
        #                   total_timesteps=num_timesteps)

        model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                           lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                           ent_coef=0.0,
                           lr=3e-4,
                           cliprange=0.2,
                           total_timesteps=num_timesteps)

    tf.reset_default_graph()

import environments.obstacle_car.params as params
import itertools

def main():
    args = mujoco_arg_parser().parse_args()

    Rs = [100, 200, 400]
    ds = [0.000, 0.005]
    obs = [0, 4, 8]

    for R, d, o in itertools.product(Rs, ds, obs):
        print("training on {}, {}, {}".format(R, d, o))
        params.R = R
        params.screen_size = (R, R)
        params.num_obstacles = o
        params.distance_rescale = R / 4  # only used in radial environment
        params.x_tolerance = R / 4

        params.reward_distance = d

        logger.configure(dir="/tmp/car_dist{}_rew{}_obs{}".format(R, d, o))
        train(params, num_timesteps=int(50000), seed=args.seed)

    Rs = [400, 800]
    ds = [0.000, 0.005]
    obs = [8, 12]

    for R, d, o in itertools.product(Rs, ds, obs):
        print("training on {}, {}, {}".format(R, d, o))
        params.R = R
        params.screen_size = (R, R)
        params.num_obstacles = o
        params.distance_rescale = R / 4  # only used in radial environment
        params.x_tolerance = R / 4

        params.reward_distance = d

        logger.configure(dir="/tmp/car_dist{}_rew{}_obs{}_longer".format(R, d, o))
        train(params, num_timesteps=int(500000), seed=args.seed)



if __name__ == '__main__':
    main()
