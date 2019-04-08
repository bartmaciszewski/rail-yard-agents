import sys
import gym
import random
import numpy as np
import tensorflow as tf

EMPTY = 0
FULL = 1
NUMBER_OF_TRACKS = 3
NUMBER_OF_CARS = 2
INBOUND_TRACK_ID = 2
OUTBOUND_TRACK_ID = 3
SWITCH_POS_A = 0
SWITCH_POS_B = 1

class RailYardEnv2(gym.Env):
    metadata = {'render.modes': ['human']}
        
    def __init__(self):
        #actions are backwad and forward       
        self.action_space = DiscreteDynamic(NUMBER_OF_TRACKS*NUMBER_OF_TRACKS*NUMBER_OF_CARS)

    def reset(self):
        self.lead1 = Track(1,5)
        self.spur1 = Track(2,5)
        self.spur2 = Track(3,5)
        self.lead1.connect(self.spur1)
        self.lead1.connect(self.spur2)
        #self.switch1 = Switch(self.lead1, self.spur1, self.spur2)
        self.tracks = {1 : self.lead1, 2 : self.spur1, 3 : self.spur2}
        self.period = 0
        for i in range(NUMBER_OF_CARS):
            self.spur1.push(RailCar(EMPTY))
        self.action_space.available_actions = self.possible_actions()

    #follow an action to transition to the next state
    def step(self,action):
        self.period += 1
        
        decoded_action = self.decode_action(action)
        
        #switch i cars from source to destination track
        for i in range(decoded_action[2]):
            self.tracks[decoded_action[1]].push(self.tracks[decoded_action[0]].pop())
        
        self.action_space.available_actions = self.possible_actions()

        #have we moved all the cars to the outbound track?
        done = True if self.tracks[OUTBOUND_TRACK_ID].number_of_cars() == NUMBER_OF_CARS else False

        return self.tracks, 10, done, None
        
    #return all the possible actions in this state
    def possible_actions(self):
        actions = []
        for source_track in self.tracks.values(): #for each track in the rail yard
            for destination_track in source_track.connected_tracks: #for all the connected tracks
                if not source_track.isEmpty() and not destination_track.isFull(): #check the source track is not empty and destination not full
                    for num_cars_to_move in range(min(source_track.number_of_cars(), destination_track.number_of_empty_spots())):
                        actions.append(self.encode_action(source_track.ID, destination_track.ID, num_cars_to_move + 1)) #encode action to move k cars
        return actions

    #encode an action to move cars from one track to another as an intetger
    def encode_action(self, source_track, destination_track, num_cars):
        i = source_track - 1
        i *= NUMBER_OF_TRACKS
        i += destination_track - 1
        #i *= NUMBER_OF_TRACKS
        i *= NUMBER_OF_CARS
        i += num_cars - 1
        #i *= NUMBER_OF_CARS
        return i

    #decode an action to move cars from one track to another
    def decode_action(self, i):
        out = []
        out.append(i % NUMBER_OF_CARS + 1)
        i = i // NUMBER_OF_CARS
        out.append(i % NUMBER_OF_TRACKS + 1) 
        i = i // NUMBER_OF_TRACKS
        out.append(i + 1)
        #assert 0 <= i < NUMBER_OF_TRACKS
        return list(reversed(out))

    def render(self,mode="human"):
        for track in self.tracks.values():
            sys.stdout.write(str(track) + "\n")
        sys.stdout.write("\n")

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

class RailCar:
    def __init__(self, empty_full):
        self.empty_or_full = empty_full

    def __str__(self):
        return str(self.empty_or_full)

class Track:
    def __init__(self, ID, length):
        self.ID = ID
        self.cars = []
        self.length = length
        self.connected_tracks = []

    def push(self,car):
        self.cars.append(car)

    def pop(self):
        return self.cars.pop()

    def isEmpty(self):
        return self.cars == []

    def isFull(self):
        return True if self.number_of_cars() == self.length else False

    def number_of_cars(self):
        return len(self.cars)

    def number_of_empty_spots(self):
        return self.length - self.number_of_cars()

    def __str__(self):
        return str([str(car) for car in self.cars])

    def connect(self, track):
        self.connected_tracks.append(track)
        track.connected_tracks.append(self)

#A switch is a track of size 1, track C diverges from the A and B straight ahead
class Switch(Track):
    def __init__(self, track_A, track_B, track_C):
        self.track_A = track_A
        self.track_B = track_B
        self.track_C = track_C
        self.A_B = SWITCH_POS_A
        super(self.__class__, self).__init__(1)
    
    #throw the switch to either A or B position
    def switch(self, switch_pos):
        self.A_B = switch_pos

class Rack(Track):
    def load(car,period):
        pass
    
    def isLoading(period):
        pass

class SimpleOneByOnePolicyAgent:
    def __init__(self,rail_yard):
        self.rail_yard = rail_yard

    def play(self):
        action = [[0 for j in range(NUMBER_OF_TRACKS)] for i in range(NUMBER_OF_TRACKS)]
        for i in reversed(range(NUMBER_OF_TRACKS-1)):
            if not self.rail_yard.tracks[i].isEmpty() and not self.rail_yard.tracks[i+1].isFull():
                action[i][i+1] = 1
        return action


#Main
rail_yard = RailYardEnv2()
#ss = SimpleOneByOnePolicyAgent(rail_yard)
done = False
rail_yard.reset()
rail_yard.render()
while not done:
    #action = ss.play()
    observation, reward, done, info = rail_yard.step(rail_yard.action_space.sample())
    rail_yard.render()