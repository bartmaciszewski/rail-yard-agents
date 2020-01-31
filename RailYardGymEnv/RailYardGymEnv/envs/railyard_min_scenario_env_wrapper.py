import gym
import RailYardGymEnv.envs.railyard_gym_env as ryge
from RailYardGymEnv.envs.railyard_model import RailYardMinScenario
from RailYardGymEnv.envs.railyard_gym_env import DiscreteDynamic, RailCarBoxSpace

class RailYardMinScenarioEnvWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        #create a simple rail yard with 3 tracks and 1 car to load
        self.env.ry = RailYardMinScenario()

        #number of actions is defined by how many combinations of cars we can move from track to track and do nothing action     
        self.env.action_space = DiscreteDynamic(self.env.ry.NUMBER_OF_TRACKS*self.env.ry.NUMBER_OF_TRACKS*self.env.ry.NUMBER_OF_CARS+1)
        
        #state is the location and state of each rail car
        self.env.observation_space = RailCarBoxSpace(self.env.ry)

    def reset(self):
        self.env.reset()
        self.env.ry = RailYardMinScenario()

        #build initial action space for this starting yard configuration
        self.env.action_space.available_actions = self.env.possible_actions()
        
        return self.env.observation_space.current_observation(self.env.ry.cars, self.env.ry.tracks, self.env.ry.loading_schedule)

if __name__ == '__main__':
    rail_yard_env = RailYardMinScenarioEnvWrapper(gym.make("RailYardGymEnv-v0"))
    rail_yard_env.reset()
    print("\nStarting game...\n")
    print("Loading Schedule:")
    print(str(rail_yard_env.ry.loading_schedule) + "\n")
    print(rail_yard_env.render())
    done = False
    while not done:    
        #get input from user
        from_track = input("From track: ")
        to_track = input("To track: ")
        num_cars = input("Number of cars: ")
        if from_track == "" or to_track  == "" or num_cars == "" :
            action = ryge.DO_NOTHING_ACTION
        else:    
            action = rail_yard_env.encode_action(int(from_track), int(to_track), int(num_cars))
            if action not in rail_yard_env.action_space.available_actions:
                action = ryge.DO_NOTHING_ACTION #user chose an unavailable action
        
        observation, reward, done, info = rail_yard_env.step(action)
        print(rail_yard_env.render())