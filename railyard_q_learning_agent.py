import RailYardGymEnv
import numpy as np
from tf_agents.environments import suite_gym

def exploration_policy(env):
    return np.random.choice(env.possible_actions())

def observation_to_str(state):
    #TODO: generalize for more than one line in observation 
    return "".join(str(e) for e in state[0])

#hyperparamerts
num_iterations = 1000
alpha0 = 0.5
decay = 0.005
gamma = 1.0

env_name = 'RailYardGymEnv-v0'
train_py_env = suite_gym.load(env_name)

#init q values
Q_values = {}

initial_state = train_py_env.reset()
initial_state_str = observation_to_str(initial_state.observation)
current_state = initial_state
for i in range(num_iterations):
    #current_state = train_py_env.observation_space.current_observation(
    #    train_py_env.ry.cars,train_py_env.ry.tracks, train_py_env.ry.loading_schedule)

    current_state_str = observation_to_str(current_state.observation)

    #initialize the q value table if this is the first time we have visited this state
    if current_state_str not in Q_values:
        Q_values[current_state_str] = [0 for _ in train_py_env.possible_actions()]

    #choose an action and explore the environment
    action = exploration_policy(train_py_env)
    action_index = train_py_env.possible_actions().index(action)
    next_state = train_py_env.step(action)    

    next_state_str = observation_to_str(train_py_env.current_time_step().observation)

    #if we have never visited the next state assume reward 0
    #otherwise max from what we saw in the past
    if next_state_str not in Q_values:
        next_value = 0
    else:
        next_value = np.max(Q_values[next_state_str])
    
    #update the Q value based on the reward we got
    alpha = alpha0 / (1 + i * decay)
    Q_values[current_state_str][action_index] *= (1 - alpha)
    Q_values[current_state_str][action_index] += alpha*(next_state.reward + gamma*next_value)
    
    if next_state.is_last():
        train_py_env.reset() 
    
    current_state = train_py_env.current_time_step()

    print("\r Iteration:{} Q value max{}".format(
        i, np.max(Q_values[initial_state_str]), end=""))
    #print(Q_values)

print(Q_values)