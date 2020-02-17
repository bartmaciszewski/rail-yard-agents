from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils.common import function
from tf_agents.eval.metric_utils import log_metrics
import logging
import base64
import imageio
import datetime

import RailYardGymEnv
from min_scenario_policy import MinYardScenarioPolicy

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

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

def train_agent(n_iterations):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/dqn_agent/' + current_time + '/train'
    writer = tf.summary.create_file_writer(train_log_dir)
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    iterator = iter(dataset)
    with writer.as_default():
        for iteration in range(n_iterations):
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            #log metrics
            print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
            if iteration % 1000 == 0:
                log_metrics(train_metrics)
                tf.summary.scalar("number_of_episodes", train_metrics[0].result(), iteration)
                tf.summary.scalar("environment_steps", train_metrics[1].result(), iteration)
                tf.summary.scalar("average_return", train_metrics[2].result(), iteration)
                tf.summary.scalar("average_episode_length", train_metrics[3].result(), iteration)
                writer.flush()

tf.compat.v1.enable_v2_behavior()

#training parameters
num_iterations = 200000#500000 #number of training iterations (e.g. play a number of steps and then train) 
collect_steps_per_iteration = 1 #how many steps to play in each training iteration
pretrain_steps = 10000 #number of steps to initialize the buffer with a pre trained policy
replay_buffer_max_length = 10000
batch_size = 32
learning_rate = 2.5e-3 #2.5e-3 how fast to update the Q value function
initial_e = 0.5 #initial epsilon
final_e = 0.01 #final epsilon
target_update_period = 100
#log_interval = 2 

#how many episodes to play to evaluate the agent 
num_eval_episodes = 5 

#create training and evaluation environment and convert to Tensorflow environments
#env_name = 'RailYardGymEnv-v0'
env_name = 'MinScenarioRailYardGymEnv-v0'
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
train_env.reset()
eval_env.reset()

#py environment used by DQN agent's observation_action_splitter when choosing next action based on state
current_py_env = train_py_env

#Build the Qnetwork
#TODO(bmac): normalize the input to the network
#preprocessing_layer = keras.layers.Lambda(
#                          lambda obs: tf.cast(obs, np.float32) / 255.)
fc_layer_params = [32]
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    #preprocessing_layer=preprocessing_layer,
    #preprocessing_combiner=tf.keras.layers.Concatenate(),
    #        [tf.keras.layers.Input(shape=(6,), dtype='int32') for _ in range(len(train_env.observation_spec()))],
    #        axis=-1),
    fc_layer_params=fc_layer_params)

train_step = tf.Variable(0,dtype=tf.int64)
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_e, # initial ε
    decay_steps=num_iterations,
    end_learning_rate=final_e) # final ε
agent = DqnAgent(train_env.time_step_spec(),
                 train_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 #observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                 target_update_period=tartget_update_period,
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

# A buffer to stores the experience from interacting with the environment
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

#An observer to write the trajectories to the buffer
replay_buffer_observer = replay_buffer.add_batch

#Training metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric()
]
logging.getLogger().setLevel(logging.INFO)

#Create the driver
collect_driver = DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=collect_steps_per_iteration) # collect # steps for each training iteration

#Collect some inital experience with random policy
initial_collect_policy = MinYardScenarioPolicy(train_env.time_step_spec(),train_env.action_spec())#RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())
#initial_collect_policy = RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

init_driver = DynamicStepDriver(
    train_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch,ShowProgress(pretrain_steps)],
    num_steps=pretrain_steps)
final_time_step, final_policy_state = init_driver.run()

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

current_py_env = train_py_env
train_agent(num_iterations)

current_py_env = eval_py_env
create_policy_eval_text_log(agent.policy, "trained-agent",num_eval_episodes)
#create_policy_eval_video(agent.policy, "trained-agent")