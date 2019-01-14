import sys
import gym
import random
import numpy as np
import tensorflow as tf

BACKWARD = 0
FORWARD = 1
NUM_EPOCHS = 500 #Each epoch is a training iteration and 2500 essentially optimizes the network
NUM_EPISODES = 1

class Truck:
    inventory = 0
    
    def load(self):
        self.inventory = 1

class LoadimgRack:
    inventory = 100
    
    def load(self, truck):
        truck.load()
        self.inventory -= 1
        print("Loaded truck. Remaining inventory:" + self.inventory)

class Locomotive:
    def __init__(self, track, position):
        self.current_track = track
        self.current_position = position
        self.current_track[self.current_position] = self

    #Move the locomotive
    def move(self, action):
        self.current_track[self.current_position] = None
        #Backward
        if action == BACKWARD and self.current_position > 0:
            self.current_position -= 1
        #Forward
        elif action == FORWARD and self.current_position < len(self.current_track) - 1:
            self.current_position += 1
        self.current_track[self.current_position] = self

    #Reset to absolute position
    def reset_to_position(self, position):
        self.current_track[self.current_position] = None
        self.current_position = position
        self.current_track[self.current_position] = self

    """
    #returms possible locomotove moves from current position
    def possibleMoves(self):
        possible_moves = []
        if self.current_position > 0:
            #left
            possible_moves.append([self.current_track, 0])
        if self.current_position < len(self.current_track) - 1:
            #right
            possible_moves.append([self.current_track, 1])
        return possible_moves
    """

class RailYardEnv(gym.Env):
    
    def __init__(self):
        #actions are backwad and forward       
        self.action_space = DiscreteDynamic(2)
        #states are all the possible positions of the locomotive
        self.observation_space = gym.spaces.Discrete(10)
        #rebuild the track and set the loco in the starting position
        self.track = [None]*10
        self.loco = Locomotive(self.track, 0)

    def reset(self):
        self.loco.reset_to_position(0)
        self.action_space.disable_actions([BACKWARD])

    #move along the track until reach the end and update the available actions at each step
    def step(self, action):
        self.loco.move(action)
        if self.loco.current_position == len(self.track) - 1:
            self.action_space.disable_actions([FORWARD])
            reward = 1000
            done = True
        elif self.loco.current_position == 0:
            self.action_space.disable_actions([BACKWARD])
            reward = -10
            done = False
        else:
            self.action_space.enable_actions([BACKWARD,FORWARD])
            reward = -10
            done = False
        #print(self.track)
        return self.loco.current_position, reward , done, None

    def render(self,mode="human"):
        out = []
        for t in self.track:
            if t == None:
                out.append(0)
            else:
                out.append(1)
        #sys.stdout.write(str(out) + "\n")
        return out

class DiscreteDynamic(gym.spaces.Discrete):

    def __init__(self, max_space):
        self.n = max_space

        #initially all actions are available
        self.available_actions = range(0, max_space)

    def disable_actions(self, actions):
        self.available_actions = [action for action in self.available_actions if action not in actions]
        return self.available_actions

    def enable_actions(self, actions):
        for action in actions:
            if action not in self.available_actions:
                self.available_actions.append(action)
        return self.available_actions

    def sample(self):
        return np.random.choice(self.available_actions)

    def contains(self, x):
        return x in self.available_actions

    @property
    def shape(self):
        return ()

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r.tolist()

print("Starting train simulator")

#initialize tensors
observations = tf.placeholder(dtype=tf.float32, shape=[None,10])
actions = tf.placeholder(dtype=tf.int64,shape=[None])
rewards = tf.placeholder(dtype=tf.float32,shape=[None])

#setup model
Y = tf.layers.dense(observations,10,activation=tf.nn.relu)
Ylogits = tf.layers.dense(Y,2)
sample_op = tf.multinomial(logits=tf.reshape(Ylogits,shape=(1,2)), num_samples=1)

#loss function
#print(Ylogits.get_shape())
#print(tf.one_hot(actions,2))
cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions,2),logits=Ylogits)
loss = tf.reduce_sum(rewards * cross_entropies)

#training operation setup
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.99)
train_op = optimizer.minimize(loss)

#trainimg iteration
with tf.Session() as sess:

    #initialize variables and game experience memory
    sess.run(tf.global_variables_initializer())
    avg_reward = [0]*NUM_EPOCHS
    
    for epoch in range(NUM_EPOCHS):

        print('\nStarting epoch ' + str(epoch) + '\n')
        epoch_memory = []
        episode_memory = []

        for episode in range(NUM_EPISODES):

            #print('\nStarting episode ' + str(episode) + '\n')
            #rail yard environment set up
            env = RailYardEnv()
            env.reset()
            counter = 0
            reward = 0
            done = False

            #run the game
            while not done:
                observation = env.render()
                #decide move tp play
                action = sess.run(sample_op, feed_dict={observations: [observation]})
                #state, step_reward, done, info = env.step(env.action_space.sample())
                state, step_reward, done, info = env.step(action)
                avg_reward[epoch] += step_reward
                #collect results
                episode_memory.append((observation, action[0][0], step_reward))
        
            #discount rewards
            episode_observations, episode_actions, episode_rewards = zip(*episode_memory)
            processed_rewards = discount_rewards(episode_rewards,0.99)

            #append to memory
            epoch_memory.extend(zip(episode_observations, episode_actions, processed_rewards))

        #training step
        epoch_observations, epoch_actions, epoch_rewards = zip(*epoch_memory)
        feed_dict = {observations: epoch_observations, actions: epoch_actions, rewards: epoch_rewards}
        sess.run(train_op, feed_dict=feed_dict)
        avg_reward[epoch] /= NUM_EPISODES

    #print stats
    for epoch in range(NUM_EPOCHS):
        print("Epoch:" + str(epoch) + " avg reward: " + str(avg_reward[epoch]))