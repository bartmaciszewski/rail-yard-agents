import sys
import gym
import random
import numpy as np
from RailYardGymEnv.envs.railyard_model import RailYard
from RailYardGymEnv.envs.railyard_model import RailYardMinScenario
from RailYardGymEnv.envs.railyard_model import Rack

from gym import spaces

DO_NOTHING_ACTION = 0

#Rewards
NEGATIVE_STEP_REWARD = -10
EPISODE_SUCCESS_REWARD = 1000

#Time limit for each loading day
MAX_NUMBER_OF_PERIODS = 100

# Dimensions of the map
MAP_WIDTH, MAP_HEIGHT = 24, 24

class RailYardGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
        
    def __init__(self):
        
        #create the rail yard objects
        self.ry = RailYard()
        #self.ry = RailYardMinScenario()

        #number of actions is defined by how many combinations of cars we can move from track to track and do nothing action     
        self.action_space = DiscreteDynamic(self.ry.NUMBER_OF_TRACKS*self.ry.NUMBER_OF_TRACKS*self.ry.NUMBER_OF_CARS+1)
        
        #state is the location and state of each rail car
        self.observation_space = RailCarBoxSpace(self.ry)
        #self.observation_space = RailCarTuplesSpace()
        #self.observation_space = spaces.Tuple([spaces.Tuple((spaces.Discrete(NUMBER_OF_TRACKS),  #which track
        #                                        spaces.Discrete(MAX_TRACK_LENGTH),  #which position
        #                                        spaces.Discrete(2),  #loaded or not
        #                                        spaces.Discrete(NUMBER_OF_SETS),  #which set for loading
        #                                        spaces.Discrete(len(PRODUCTS))))  #which product to load
        #                                       for _ in range(NUMBER_OF_CARS)])
        #print(self.observation_space.sample())
        #self.observation_space = spaces.MultiDiscrete([NUMBER_OF_CARS,   #which car
        #                                            NUMBER_OF_TRACKS,  #which track
        #                                            MAX_TRACK_LENGTH,  #which position
        #                                            2,  #loaded or not
        #                                            NUMBER_OF_SETS,  #which set for loading
        #                                            len(PRODUCTS)])  #which product to load
                                               
    def reset(self):
        self.period = 0
        
        #rebuild the rail yard
        self.ry = RailYard()
        #self.ry = RailYardMinScenario()

        #build initial action space for this starting yard configuration
        self.action_space.available_actions = self.possible_actions()
        
        return self.observation_space.current_observation(self.ry.cars, self.ry.tracks, self.ry.loading_schedule)

    def step(self, action):
        """Follow an action to transition to the next state of the yard."""
        done = False

        #step 1: check if we have ran out of time
        if self.period == MAX_NUMBER_OF_PERIODS:
            done = True
            return self.observation_space.current_observation(self.ry.cars, self.ry.tracks, self.ry.loading_schedule), NEGATIVE_STEP_REWARD, done, None
 
        #step 2: continue loading any racks
        for rack in self.ry.racks.values():
            if rack.is_currently_loading():
                rack.load_step()

        #make sure the action is valid for the state
        if self.action_space.contains(action):

            if action != DO_NOTHING_ACTION:
                decoded_action = self.decode_action(action)
            
                #step 3: switch i cars from source to destination track
                for i in range(decoded_action[2]):
                    self.ry.tracks[decoded_action[1]].push(self.ry.tracks[decoded_action[0]].pop())
            
                #step 4: start loading if the destination was a rack
                if isinstance(self.ry.tracks[decoded_action[1]],Rack):
                    self.ry.tracks[decoded_action[1]].start_load()

        #step 5: determine the next possible actions from this new state
        self.action_space.available_actions = self.possible_actions()

        #have we moved all the cars to the outbound track?
        done = self.is_success_state() 
  
        #reward completion of the game and penalize every move
        reward = NEGATIVE_STEP_REWARD if not done else EPISODE_SUCCESS_REWARD

        self.period += 1
        
        return self.observation_space.current_observation(self.ry.cars, self.ry.tracks, self.ry.loading_schedule), reward, done, None
    
    def is_success_state(self):
        """Returns True if we have loaded all the required cars and moved them to the outbound track."""

        #Check if all scheduled cars loaded
        for i in range(self.ry.loading_schedule.number_of_sets()):
            for car_to_load in self.ry.loading_schedule.get_cars(i+1):
                car_loaded = False
                #check if car is not empty and is on the outbound
                if not car_to_load.is_empty():
                    for car_on_outbound in self.ry.outbound.cars:
                        if car_on_outbound == car_to_load:
                            car_loaded = True
                #return fals if there is car that has not been loaded or placed on the outbound
                if car_loaded == False:
                    return False
        return True

    def possible_actions(self):
        """Return all the possible actions in this state."""
        actions = []
        actions.append(DO_NOTHING_ACTION)
        for source_track in self.ry.tracks.values(): #for each track in the rail yard
            for destination_track in source_track.connected_tracks: #for all the connected tracks
                if not source_track.derail_up() and not destination_track.derail_up(): #don’t move any cars to/from loading racks if derail is up (e.g. loading)
                    if not source_track.is_empty() and not destination_track.is_full(): #check the source track is not empty and destination not full
                        for num_cars_to_move in range(min(source_track.number_of_cars(), destination_track.number_of_empty_spots())):
                            actions.append(self.encode_action(source_track.ID, destination_track.ID, num_cars_to_move + 1)) #encode action to move k cars
        return actions
 
    #encode an action to move cars from one track to another as an intetger
    def encode_action(self, source_track, destination_track, num_cars):
        i = source_track - 1
        i *= self.ry.NUMBER_OF_TRACKS
        i += destination_track - 1
        #i *= NUMBER_OF_TRACKS
        i *= self.ry.NUMBER_OF_CARS
        i += num_cars - 1
        #i *= NUMBER_OF_CARS
        return i

    #decode an action to move cars from one track to another
    def decode_action(self, i):
        out = []
        out.append(i % self.ry.NUMBER_OF_CARS + 1)
        i = i // self.ry.NUMBER_OF_CARS
        out.append(i % self.ry.NUMBER_OF_TRACKS + 1)
        i = i // self.ry.NUMBER_OF_TRACKS
        out.append(i + 1)
        #assert 0 <= i < NUMBER_OF_TRACKS
        return list(reversed(out))

    def render(self,mode="human"):
        output = "Period: " + str(self.period) + "\n"
        #sys.stdout.write("Period: " + str(self.period) + "\n")
        for track in self.ry.tracks.values():
            if isinstance(track, Rack):
                #sys.stdout.write("Rack " + str(track) + "\n")
                output += "Rack " + str(track) + "\n"
            else:
                #sys.stdout.write("Track " + str(track) + "\n")
                output += "Track " + str(track) + "\n"
        #sys.stdout.write("\n")
        return output


class DiscreteDynamic(gym.spaces.Discrete):
    
    def __init__(self, max_space):
        #initially all actions are available
        self.available_actions = range(0, max_space)
        self.max_space = max_space
        super(DiscreteDynamic, self).__init__(max_space)

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

    def action_space_mask(self):
        """Creates a boolean mask for all possible actions in the current state

        Returns:
            [1,0,1,...] where 1 if the nth action is currently valid and 0 otherwise
        """
        mask = [0 for _ in range(0, self.max_space)]
        for action in self.available_actions:
            mask[action] = 1
        return mask

    #@property
    #def shape(self):
    #    return ()


class RailCarTuplesSpace(gym.spaces.Tuple):
    """
    An observation space that represents the rail yard state as a tuple of tuples for each rail car.
    """
   
    def __init__(self):
        """
        Creates a new observation space as a set of tuples for all cars
        """
        super(RailCarTuplesSpace, self).__init__([spaces.Tuple((spaces.Discrete(NUMBER_OF_TRACKS),  #which track
                                                spaces.Discrete(MAX_TRACK_LENGTH),  #which position
                                                spaces.Discrete(2),  #loaded or not
                                                spaces.Discrete(NUMBER_OF_SETS),  #which set for loading
                                                spaces.Discrete(len(PRODUCTS))))  #which product to load
                                                for _ in range(NUMBER_OF_CARS)])
   
    def current_observation(self, cars, tracks, loading_schedule):
        """
        Represents the current state of the yard as a tuple of tuples for each car
        
        Args:
            cars : List of rail cars in the yard
            tracks : A Dictionary of all the tracks in the yard
            loading_schedule : The loading schedule

        Returns:
            ( (track, position on track, loaded or empty, set number and product if on loading schedule),
            ...
            (...,...,...,...) )
        """
    
        observation = [None]*len(cars)
        #for each car on each track
        for track in tracks.values():
            
            car_position = 0
            for car in track.get_cars():

                #determine the set and product if on load schedule
                found_car_on_schedule = False
                for set in range(loading_schedule.number_of_sets()):
                    for product in PRODUCTS.keys():
                
                        #car is on schedule so add the set and product to the observation
                        if loading_schedule.is_on_set_schedule(car, set+1, PRODUCTS[product]) == True:
                            observation[car.ID] = (track.ID, car_position, car.empty_or_full, set+1, product)
                            found_car_on_schedule = True
                
                #car is not on schedule so
                if found_car_on_schedule == False:
                    observation[car.ID] = (track.ID, car_position, car.empty_or_full, 0, 0)
                
                car_position += 1

        return tuple(observation)



class RailCarBoxSpace(gym.spaces.Box):
    """
    An observation space that represents the rail yard state as a 2 dimensional array
    with a row for each car that has its track, position, load state, and whether
    it was on the schedule.
    """

    def __init__(self,rail_yard):
        """
        Creates a new observation space as a 2 dimensional box of size (# cars, # variables about each car)

        Args:
            rail_yard : a handle to a RailYard object (cars, tracks, etc)
        """
        self.ry = rail_yard
        #how many fields to represent state of each car
        self.NUM_CAR_VARIABLES = 5 
        #what is the biggest possible field value
        self.MAX_CAR_VARIABLE_VALUE = max(self.ry.MAX_TRACK_LENGTH, self.ry.NUMBER_OF_CARS, self.ry.NUMBER_OF_CARS, self.ry.NUMBER_OF_SETS, self.ry.NUMBER_OF_TRACKS, len(self.ry.PRODUCTS))
        super(RailCarBoxSpace, self).__init__(0,self.MAX_CAR_VARIABLE_VALUE, [self.ry.NUMBER_OF_CARS, self.NUM_CAR_VARIABLES], dtype=np.int32)
                                               
    def current_observation(self, cars, tracks, loading_schedule):
        """
        Represents the current state of the yard as a 2 dimensional box with info for each car as a row of ints
        
        Args:
            cars : List of rail cars in the yard
            tracks : A Dictionary of all the tracks in the yard
            loading_schedule : The loading schedule

        Returns:
            ( (track, position on track, loaded or empty, set number and product if on loading schedule),
            ...
            (...,...,...,...) )
        """
    
        observation = [[None]*self.NUM_CAR_VARIABLES for _ in range(len(cars))]
        #for each car on each track
        for track in tracks.values():
            
            car_position = 0
            for car in track.get_cars():

                #determine the set and product if on load schedule
                found_car_on_schedule = False
                for set in range(loading_schedule.number_of_sets()):
                    for product in self.ry.PRODUCTS.keys():
          
                        #car is on schedule so add the set and product to the observation
                        if loading_schedule.is_on_set_schedule(car, set+1, self.ry.PRODUCTS[product]) == True:
                            observation[car.ID][0] = track.ID
                            observation[car.ID][1] = car_position
                            observation[car.ID][2] = car.empty_or_full
                            observation[car.ID][3] = set+1
                            observation[car.ID][4] = product
                            found_car_on_schedule = True
                
                #car is not on schedule so
                if found_car_on_schedule == False:
                    observation[car.ID][0] = track.ID
                    observation[car.ID][1] = car_position
                    observation[car.ID][2] = car.empty_or_full
                    observation[car.ID][3] = 0
                    observation[car.ID][4] = 0                
                
                car_position += 1

        return np.array(observation)

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

class MyopicGreedySortByProductPolicy(RailyardPolicy):
    """A policy that moves cars to the marshalling tracks by product as the first sorting step.  
        At every step it first looks to see if there are any cars finished loading that can be moved to the outbound,
        then if there are any cars that can be moved under a rack from the marshalling tracking, 
        then if there are any cars on the inbound that can be moved to the marshalling track.  
        
        Note this policy does not respect sets nor the loading schedule - it greedily loads cars and pushes the first ones loaded to the outbound 
        Therefore the policy is not guaranteed to meet the daily loading schedule.
        It does not consider multiple periods (myopic)

    Attributes:
        rail_yard: the reference to the rail yard
        inbound: the inbound track
        outbound: the outbound track
        racks: the loading racks
        marshalling_tracks: the marshalling tracks to sort by product 
    """

    def __init__(self, rail_yard, inbound, outbound, racks, marshalling_tracks):
        self.rail_yard = rail_yard
        self.inbound = inbound
        self.outbound = outbound
        self.racks = racks
        self.marshalling_tracks = marshalling_tracks
        self.loco = rail_yard.loco

    def next_action(self):
        
        if self.loco.is_active():
            next_hop = self.loco.next_hop()
            return self.rail_yard.encode_action(next_hop[0], next_hop[1], next_hop[2])
        else:
            #are there any racks that can be freed up?
            for rack in self.racks.values():
                if not rack.is_currently_loading() and not rack.is_empty():
                    #move the maximum number of cars from the rack finished loading to the outbound
                    #return self.rail_yard.encode_action(rack.ID, self.outbound.ID, min(rack.number_of_cars(), self.outbound.number_of_empty_spots()))
                    self.loco.move_cars(self.rail_yard.tracks, rack, self.outbound, min(rack.number_of_cars(), self.outbound.number_of_empty_spots()))
                    next_hop = self.loco.next_hop()
                    return self.rail_yard.encode_action(next_hop[0], next_hop[1], next_hop[2])

            #are there any cars ready on the marshalling track that can be loaded?
            for marshalling_track1 in self.marshalling_tracks:
                if not marshalling_track1.is_empty():
                    #are there any empty racks that can load this product
                    for rack2 in self.racks.values():
                        if rack2.is_empty() and rack2.product == marshalling_track1.cars[0].product:
                            #move the maximum number of cars from the marshalling track to the rack
                            #return self.rail_yard.encode_action(marshalling_track1.ID, rack2.ID, min(marshalling_track1.number_of_cars(), rack2.number_of_empty_spots()))
                            self.loco.move_cars(self.rail_yard.tracks, marshalling_track1, rack2, min(marshalling_track1.number_of_cars(), rack2.number_of_empty_spots()))
                            next_hop = self.loco.next_hop()
                            return self.rail_yard.encode_action(next_hop[0], next_hop[1], next_hop[2])

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
                        #return self.rail_yard.encode_action(self.inbound.ID, marshalling_track2.ID, min(count_same_product, marshalling_track2.number_of_empty_spots()))
                        self.loco.move_cars(self.rail_yard.tracks, self.inbound, marshalling_track2, min(count_same_product, marshalling_track2.number_of_empty_spots()))
                        next_hop = self.loco.next_hop()
                        return self.rail_yard.encode_action(next_hop[0], next_hop[1], next_hop[2])

        return DO_NOTHING_ACTION

class MyopicSortBySetPolicy(RailyardPolicy):
    """
    UNDER DEVELOPMENT

    A policy that loads cars and puts them on the outbound one set at a time

    Attributes:
        rail_yard: the reference to the rail yard
        inbound: the inbound track
        outbound: the outbound track
        racks: the loading racks
        marshalling_tracks: the marshalling tracks to sort by set (need one for each set)
    """
    def __init__(self, rail_yard, inbound, outbound, racks, marshalling_tracks):
        self.rail_yard = rail_yard
        self.inbound = inbound
        self.outbound = outbound
        self.racks = racks
        self.marshalling_tracks = marshalling_tracks
        self.loco = rail_yard.locomotive
        self.num_sets = rail_yard.loading_schedule.number_of_sets()
        self.current_set = 1

    def next_action(self):
        if self.loco.is_active():
            next_hop = self.loco.next_hop()
            return self.rail_yard.encode_action(next_hop[0], next_hop[1], next_hop[2])
        else:
            if self.current_set <= self.num_sets:                            
                #step 1: move the largest group of cars in the set from inbound to outbound - UNDER DEVELOPMENT
                cars_to_move = 0
                found_car_in_set = False
                for car in self.inbound.get_cars():
                    #if the car is in the set then move it and keep looking at the rest of cars
                    if car in self.rail_yard.loading_schedule.get_cars(self.current_set):
                        found_car_in_set = True
                        cars_to_move += 1
                    #car we are looking at is not in set but we have already found one then we should move all cars prior to this one
                    elif found_car_in_set:
                        self.loco.move_cars(self.rail_yard.tracks, self.inbound, self.marshalling_tracks[0], cars_to_move)
                        next_hop = self.loco.next_hop()
                        return self.rail_yard.encode_action(next_hop[0], next_hop[1], next_hop[2])
                    #car is not in set and we haven’t found one yet so keep on looking
                    else:
                        cars_to_move += 1

                #step 2: move each car for the set under the correct rack - UNDER DEVELOPMENT

                #step 3: move loaded cars from rack to outbound - UNDER DEVELOPMENT

        self.current_set += 1

def main():
    """ Main 
    """
    rail_yard_env = RailYardGymEnv()
    rail_yard_env.reset()
    print("\nStarting game...\n")
    print("Loading Schedule:")
    print(str(rail_yard_env.ry.loading_schedule) + "\n")
    print(rail_yard_env.render())
    #ss = SimpleOneByOnePolicyAgent(rail_yard)
    #policy = MyopicGreedySortByProductPolicy(rail_yard, rail_yard.inbound, rail_yard.outbound, rail_yard.racks, rail_yard.marshalling_tracks)
    done = False
    while not done:
        #action = policy.next_action()
        
        #get input from user
        from_track = input("From track: ")
        to_track = input("To track: ")
        num_cars = input("Number of cars: ")
        if from_track == "" or to_track  == "" or num_cars == "" :
            action = DO_NOTHING_ACTION
        else:    
            action = rail_yard_env.encode_action(int(from_track), int(to_track), int(num_cars))
            if action not in rail_yard_env.action_space.available_actions:
                action = DO_NOTHING_ACTION #user chose an unavailable action
        
        #observation, reward, done, info = rail_yard.step(rail_yard.action_space.sample())
        observation, reward, done, info = rail_yard_env.step(action)
        print(rail_yard_env.render())

if __name__ == '__main__':
    main()