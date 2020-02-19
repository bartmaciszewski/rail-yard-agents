from gym.envs.registration import register

register(
    id='RailYardGymEnv-v0',
    max_episode_steps=100,
    entry_point='RailYardGymEnv.envs.railyard_gym_env:RailYardGymEnv',
)

register(
    id='MinScenarioRailYardGymEnv-v0',
    max_episode_steps=100,
    entry_point='RailYardGymEnv.envs.railyard_gym_env:MinScenarioRailYardGymEnv',
)