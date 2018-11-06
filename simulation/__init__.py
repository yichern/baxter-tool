from gym.envs.registration import registry, register, make, spec

register(
    id='BaxterReachEnv-v1',
    entry_point='simulation.envs.baxter.reach:BaxterReachEnv',
)
