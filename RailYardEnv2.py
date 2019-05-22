import sys
import gym
import random
import numpy as np
import tensorflow as tf

EMPTY = 0
FULL = 1
NUMBER_OF_TRACKS = 5
NUMBER_OF_CARS = 4
INBOUND_TRACK_ID = 2
OUTBOUND_TRACK_ID = 5
DO_NOTHING_ACTION = 0
SWITCH_POS_A = 0
SWITCH_POS_B = 1
PRODUCTS = {"MOGAS" : "M", "DIESEL" : "D", "JET" : "J", "ASPHALT" : "A"}

class RailYardEnv2(gym.Env):
    metadata = {'render.modes': ['human']}
        
    def __init__(self):
        #actions are backwad and forward       
        self.action_space = DiscreteDynamic(NUMBER_OF_TRACKS*NUMBER_OF_TRACKS*NUMBER_OF_CARS)

    def reset(self):
        self.period = 0
        self.lead1 = Track(1,5)
        self.inbound = Track(2,5)
        self.rack1 = Rack(3,2,PRODUCTS["MOGAS"],2)
        self.rack2 = Rack(4,2,PRODUCTS["DIESEL"],2)    
        self.outbound = Track(5,5)
        self.lead1.connect(self.inbound)
        self.lead1.connect(self.rack1)
        self.lead1.connect(self.rack2)
        self.lead1.connect(self.outbound)
        #self.switch1 = Switch(self.lead1, self.spur1, self.spur2)
        self.tracks = {1 : self.lead1, 2 : self.inbound, 3 : self.rack1, 4 : self.rack2, 5 : self.outbound}
        self.racks = {1 : self.rack1, 2 : self.rack2}
        self.cars = []
        for i in range(2):
            self.cars.append(RailCar(EMPTY, PRODUCTS["MOGAS"]))
            self.inbound.push(self.cars[i])
        for i in range(2):
            self.cars.append(RailCar(EMPTY, PRODUCTS["DIESEL"]))
            self.inbound.push(self.cars[i])
        self.action_space.available_actions = self.possible_actions()

    def step(self,action):
        """Follow an action to transition to the next state of the yard."""

        #step 1: continue loading any racks
        for rack in self.racks.values():
            if rack.is_loading:
                rack.load_step()

        if action != DO_NOTHING_ACTION:
            decoded_action = self.decode_action(action)
        
            #step 2: switch i cars from source to destination track
            for i in range(decoded_action[2]):
                self.tracks[decoded_action[1]].push(self.tracks[decoded_action[0]].pop())
        
            #step 3: start loading if the destination was a rack
            if isinstance(self.tracks[decoded_action[1]],Rack):
                self.tracks[decoded_action[1]].start_load()

        #step 4: determine the next possible actions from this new state
        self.action_space.available_actions = self.possible_actions()

        #have we moved all the cars to the outbound track?
        done = self.is_success_state() 

        self.period += 1

        return self.tracks, 10, done, None
    
    def is_success_state(self):
        """Returns True if we have loaded all the required cars and moved them to the outbound track."""

        if self.tracks[OUTBOUND_TRACK_ID].number_of_cars() == NUMBER_OF_CARS:
            #All the cars are on the outbound; ensure they are all loaded
            for car in self.tracks[OUTBOUND_TRACK_ID].cars:
                return not car.is_empty() #False if at least one car is empty
            return True
        else:
            return False

    #return all the possible actions in this state
    def possible_actions(self):
        actions = []
        actions.append(DO_NOTHING_ACTION)
        for source_track in self.tracks.values(): #for each track in the rail yard
            for destination_track in source_track.connected_tracks: #for all the connected tracks
                if not source_track.derail_up() and not destination_track.derail_up(): #don’t move any cars to/from loading racks
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
        sys.stdout.write("Period: " + str(self.period) + "\n")
        for track in self.tracks.values():
            if isinstance(track, Rack):
                sys.stdout.write("Rack " + str(track) + "\n")
            else:
                sys.stdout.write("Track " + str(track) + "\n")
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
    def __init__(self, empty_full, product):
        self.empty_or_full = empty_full
        self.product = product

    def is_empty(self):
        return True if self.empty_or_full == EMPTY else False

    def __str__(self):
        if self.empty_or_full == EMPTY:
            return self.product.lower()
        else:
            return self.product.capitalize()

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

    def derail_up(self):
        return False

    def __str__(self):
        return str(self.ID) + ": " + str([str(car) for car in self.cars])

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
    def __init__(self, ID, num_bays, product, load_time):
        self.load_time = load_time
        self.product = product
        self.current_load_time = 0
        self.is_loading = False
        super(self.__class__, self).__init__(ID, num_bays)

    def start_load(self):
        """Starts the process of loading cars under a rack if at least one car is product compatible."""

        for car in self.cars:
            if car.product == self.product:
                self.current_load_time = 0
                self.is_loading = True
                break

    def load_step(self):
        """Continues loading cars on rack. Finish loading cars if sufficient time has passed."""

        if self.is_loading == True and self.current_load_time >= self.load_time:
            #we are done loading
            for car in self.cars:
                #only fill up car for the rack’s product
                if car.product == self.product:
                    car.empty_or_full = FULL 
            self.is_loading = False
            self.current_load_time = 0

        self.current_load_time += 1

    def is_loading(self):
        return self.is_loading

    def derail_up(self):
        return self.is_loading

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