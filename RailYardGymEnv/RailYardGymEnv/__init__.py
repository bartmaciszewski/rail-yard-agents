from gym.envs.registration import register

register(
    id='RailYardGymEnv-v0',
    entry_point='RailYardGymEnv.envs:RailYardGymEnv',
)