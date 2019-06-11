import sys
import gym
import random
import numpy as np
import tensorflow as tf

EMPTY = 0
FULL = 1
NUMBER_OF_TRACKS = 7
NUMBER_OF_CARS = 4
INBOUND_TRACK_ID = 2
OUTBOUND_TRACK_ID = 7
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
        
        #Create tracks
        self.lead1 = Track(1,5)
        self.inbound = Track(2,5)
        self.marshalling_track1 = MarshallingTrack(3,5,PRODUCTS["MOGAS"])
        self.marshalling_track2 = MarshallingTrack(4,5,PRODUCTS["DIESEL"])
        self.rack1 = Rack(5,2,PRODUCTS["MOGAS"],2)
        self.rack2 = Rack(6,2,PRODUCTS["DIESEL"],2)    
        self.outbound = Track(7,5)
        
        #Connect tracks to form network
        self.lead1.connect(self.inbound)
        self.lead1.connect(self.marshalling_track1)
        self.lead1.connect(self.marshalling_track2)
        self.lead1.connect(self.rack1)
        self.lead1.connect(self.rack2)
        self.lead1.connect(self.outbound)
        #self.switch1 = Switch(self.lead1, self.spur1, self.spur2)
        
        #Create track reference lists
        self.tracks = {1 : self.lead1, 2 : self.inbound, 3 :  self.marshalling_track1, 4 : self.marshalling_track2, 5 : self.rack1, 6 : self.rack2, 7 : self.outbound}
        self.racks = {1 : self.rack1, 2 : self.rack2}
        self.marshalling_tracks = [self.marshalling_track1, self.marshalling_track2]

        #Create cars
        self.cars = []
        for i in range(2):
            self.cars.append(RailCar("m" + str(i+1),EMPTY, PRODUCTS["MOGAS"]))
            self.inbound.push(self.cars[i])
        for j in range(2,4):
            self.cars.append(RailCar("d" + str(j+1),EMPTY, PRODUCTS["DIESEL"]))
            self.inbound.push(self.cars[j])
        
        #Build load schedule
        self.loading_schedule = LoadingSchedule()
        self.loading_schedule.add_to_schedule(self.cars[1], self.cars[1].product)
        self.loading_schedule.add_to_schedule(self.cars[2], self.cars[2].product)

        #build initial action space
        self.action_space.available_actions = self.possible_actions()

    def step(self,action):
        """Follow an action to transition to the next state of the yard."""

        #step 1: continue loading any racks
        for rack in self.racks.values():
            if rack.is_currently_loading():
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

        #Check if all scheduled cars loaded
        for car_to_load in self.loading_schedule.get_cars():
            car_loaded = False
            #check if car is not empty and is on the outbound
            if not car_to_load.is_empty():
                for car_on_outboun in self.outbound.cars:
                    if car_on_outboun == car_to_load:
                        car_loaded = True
            #return fals if there is car that has not been loaded or placed on the outbound
            if car_loaded == False:
                return False
        return True

    def possible_actions(self):
        """Return all the possible actions in this state."""
        actions = []
        actions.append(DO_NOTHING_ACTION)
        for source_track in self.tracks.values(): #for each track in the rail yard
            for destination_track in source_track.connected_tracks: #for all the connected tracks
                if not source_track.derail_up() and not destination_track.derail_up(): #don’t move any cars to/from loading racks
                    if not source_track.is_empty() and not destination_track.is_full(): #check the source track is not empty and destination not full
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
    def __init__(self, ID, empty_full, product):
        self.ID = ID
        self.empty_or_full = empty_full
        self.product = product

    def is_empty(self):
        return True if self.empty_or_full == EMPTY else False

    def __str__(self):
        if self.empty_or_full == EMPTY:
            return self.ID.lower()
        else:
            return self.ID.capitalize()

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

    def peek(self):
        return self.cars[-1]

    def is_empty(self):
        return self.cars == []

    def is_full(self):
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

class MarshallingTrack(Track):
    """A track that is designated for a specific product.
    
    Attributes:
        ID: the unique inteer identifier of this track in the yard
        length: the length of the track in car units
        product the type of proudct for which  this track is intended 
    """

    def __init__(self, ID, length, product):
        self.product = product
        super(self.__class__, self).__init__(ID, length)

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

    def is_currently_loading(self):
        return self.is_loading

    def derail_up(self):
        return self.is_loading

class LoadingSchedule:
    """A list of tuples that represents the rail cars and products that need to be loaded on a given day
    """
    
    def __init__(self):
        self.loading_schedule = []
        self.cars = []
    
    def add_to_schedule(self, car, product):
        self.loading_schedule.append([car, product])
        self.cars.append(car)
        
    def get_cars(self):
        return self.cars
    
    def is_on_schedule(self, car, product):
        for car_product in self.loading_schedule:
            if car_product[0] == car and car_product[1] == product:
                return True
        return False

    def __str__(self):
        print_string = ""
        for car_product in self.loading_schedule:
            print_string += "Car: " + car_product[0].ID + " Product:" + car_product[1] + "\n"
        return print_string

class RailyardPolicy:
    def __init__(self, rail_yard):
        self.rail_yard = rail_yard

    def next_action(self):
        raise NotImplementedError("Please Implement this method")

class SimpleOneByOnePolicy(RailyardPolicy):
    def next_action(self):
        action = [[0 for j in range(NUMBER_OF_TRACKS)] for i in range(NUMBER_OF_TRACKS)]
        for i in reversed(range(NUMBER_OF_TRACKS-1)):
            if not self.rail_yard.tracks[i].is_empty() and not self.rail_yard.tracks[i+1].is_full():
                action[i][i+1] = 1
        return action

class SimpleTrainSortPolicy(RailyardPolicy):
    def __init__(self, rail_yard, inbound, outbound, racks, marshalling_tracks):
        self.rail_yard = rail_yard
        self.inbound = inbound
        self.outbound = outbound
        self.racks = racks
        self.marshalling_tracks = marshalling_tracks

    def next_action(self):
        #are there any racks that can be freed up?
        for rack in self.racks.values():
            if not rack.is_currently_loading() and not rack.is_empty():
                #move the maximum number of cars from the rack finished loading to the outbound
                return self.rail_yard.encode_action(rack.ID, self.outbound.ID, min(rack.number_of_cars(), self.outbound.number_of_empty_spots()))

        #are there any cars ready on the marshalling track that can be loaded?
        for marshalling_track1 in self.marshalling_tracks:
            if not marshalling_track1.is_empty():
                #are there any empty racks that can load this product
                for rack2 in self.racks.values():
                    if rack2.is_empty() and rack2.product == marshalling_track1.cars[0].product:
                        #move the maximum number of cars from the marshalling track to the rack
                        return self.rail_yard.encode_action(marshalling_track1.ID, rack2.ID, min(marshalling_track1.number_of_cars(), rack2.number_of_empty_spots()))

        #are there any inbound cars left to sort by produc?
        if not self.inbound.is_empty():
            for marshalling_track2 in self.marshalling_tracks:
                #make sure the marshalling track has space and is intended for the product avaible at the top of the inbound
                if not marshalling_track2.is_full() and self.inbound.peek().product == marshalling_track2.product:
                    #count how many cars of the same product are at the top of track available to pull
                    count_same_product = 0
                    last_product = self.inbound.peek().product
                    for car in reversed(self.inbound.cars):
                        if last_product == car.product:
                            count_same_product += 1
                            last_product = car.product
                        else: 
                            break
                    return self.rail_yard.encode_action(self.inbound.ID, marshalling_track2.ID, min(count_same_product, marshalling_track2.number_of_empty_spots()))

        return DO_NOTHING_ACTION

#Main
rail_yard = RailYardEnv2()
rail_yard.reset()
print("\nStarting game...\n")
print("Loading Schedule:")
print(str(rail_yard.loading_schedule) + "\n")
rail_yard.render()
#ss = SimpleOneByOnePolicyAgent(rail_yard)
policy = SimpleTrainSortPolicy(rail_yard, rail_yard.inbound, rail_yard.outbound, rail_yard.racks, rail_yard.marshalling_tracks)
done = False
while not done:
    action = policy.next_action()
    #observation, reward, done, info = rail_yard.step(rail_yard.action_space.sample())
    observation, reward, done, info = rail_yard.step(action)
    rail_yard.render()