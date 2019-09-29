from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

import gym
import RailYardGymEnv
env = gym.make('RailYardGymEnv-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())

environment = suite_gym.load('RailYardGymEnv-v0')