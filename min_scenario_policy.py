"""Policy implementation that generates optimal actions for
a minimal yard scenario."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.utils import nest_utils

class MinYardScenarioPolicy(tf_policy.Base):
  """Returns optimal actions for the min yard scenario"""

  def __init__(self, time_step_spec, action_spec, *args, **kwargs):
    super(MinYardScenarioPolicy, self).__init__(
        time_step_spec, action_spec, *args, **kwargs)

  def _variables(self):
    return []

  @tf.function
  def _action(self, time_step, policy_state, seed):
    current_track = time_step.observation[0,0,0]
    is_loaded = time_step.observation[0,0,2]
    move_1_to_3 = 2
    move_2_to_1 = 3
    move_1_to_2 = 1
    move_3_to_1 = 6
    do_nothing = 0
    if current_track == 1 and is_loaded == 1:
      action_ = tf.constant(move_1_to_3,dtype=tf.int64)
    elif current_track == 2 and is_loaded == 1:
      action_ = tf.constant(move_2_to_1,dtype=tf.int64)
    elif current_track == 1 and is_loaded == 0:
      action_ = tf.constant(move_1_to_2,dtype=tf.int64)
    elif current_track == 3 and is_loaded == 0:
      action_ = tf.constant(move_3_to_1,dtype=tf.int64)
    else:
      action_ = tf.constant(do_nothing,dtype=tf.int64)

    step = policy_step.PolicyStep(action_, policy_state)

    return step

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'MinYardScenarioPolicy does not support distributions.')
