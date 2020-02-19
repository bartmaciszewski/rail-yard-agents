from gym.envs.registration import register

register(
    id='RailYardGymEnv-v0',
    max_episode_steps=10000,
    entry_point='RailYardGymEnv.envs.railyard_gym_env:RailYardGymEnv',
)