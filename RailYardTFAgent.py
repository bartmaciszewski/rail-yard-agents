from __future__ import absolute_import, division, print_function

import base64
import imageio
#import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
#import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import RailYardGymEnv

def compute_avg_return(environment, policy, num_episodes=10):
    """Computes the average return of a policy.

    Args:
        environment : a TF Python Environment
        policy : the policy to execute at each step
        num_episodes : number of times to play the environment 
    
    Returns:
        The average return of the policy over the specificed number of episodes as a float
    """
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(environment, policy, buffer):
    """Adds data to the buffer from a single state transition according to the policy.
    
    Args:
        environment : a TF Python environment
        policy : the policy to use to select an action for the step
        buffer : the buffer to which to add the transition data
    Returns:
        None
    """
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    #Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    """Collects in a buffer results of executing a policy for some number of steps.
    
    Args:
        env : a TF Python environment
        policy : the policy to use to select actions
        buffer : the buffer in which to store the results
        steps : the number of steps to execute
    Returns:
        None
    """
    for _ in range(steps):
        collect_step(env, policy, buffer)

def observation_and_action_constraint_splitter(observation):
    """Used by DQN agent to constrain the action space based on state. 
    Uses the current_py_env global variable to get the constrained action space

    Args:
        observation : the current state tensor returned by the environment
    Returns:
        (observation, action_mask) : (the current state tensor to be passed to the Q network, 
            a bit string where 1 is a valid action in the current state)

    TODO: Derive the mask based on observation rather than leveraging the environment
    """
    return observation, tf.convert_to_tensor(current_py_env.action_space.action_space_mask(),dtype=tf.int32)

def create_policy_eval_text_log(policy, filename, num_episodes=5):
    logfile = open(filename + ".txt","w")
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        logfile.write(eval_py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            logfile.write("=============================\n")
            logfile.write("Action: " + str(eval_py_env.decode_action(action_step.action.numpy()[0])) + "\n")
            logfile.write(eval_py_env.render())


def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())

tf.compat.v1.enable_v2_behavior()

#training parameters
num_iterations = 100 #number of training iterations (e.g. play a number of steps and then train) 
collect_steps_per_iteration = 100 #how many steps to play in each training iteration
replay_buffer_max_length = 10000
batch_size = 1000
learning_rate = 1e-3
log_interval = 2 

#how many episodes to play to evaluate the agent 
num_eval_episodes = 5 
eval_interval = 2

#create training and evaluation environment and convert to Tensorflow environments
env_name = 'RailYardGymEnv-v0'
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
train_env.reset()
eval_env.reset()

#py environment used by DQN agent's observation_action_splitter when choosing next action based on state
current_py_env = train_py_env
'''
#Test spaces
print('Observation Spec:')
print(train_env.time_step_spec().observation)

print('Reward Spec:')
print(train_env.time_step_spec().reward)

print('Action Spec:')
print(train_env.action_spec())
'''

#Build the Qnetwork
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    #preprocessing_combiner=tf.keras.layers.Concatenate(),
    #        [tf.keras.layers.Input(shape=(6,), dtype='int32') for _ in range(len(train_env.observation_spec()))],
    #        axis=-1),
    fc_layer_params=fc_layer_params)

#print(train_env.observation_spec())
#print('Observation Spec:')
#print(train_env.time_step_spec().observation)

#Instantiate the DQN agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
    epsilon_greedy=0.5,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()

# A buffer to stores the experience from interacting with the environment
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
current_py_env = eval_py_env
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

current_py_env = train_py_env

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)
    if train_env.current_time_step().is_last():
        train_env.reset()

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    current_py_env = eval_py_env
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
    current_py_env = train_py_env

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=1000,bottom=-10000)

current_py_env = eval_py_env
create_policy_eval_text_log(agent.policy, "trained-agent")
#create_policy_eval_video(agent.policy, "trained-agent")