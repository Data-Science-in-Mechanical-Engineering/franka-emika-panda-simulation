from gym.envs.registration import register

register(
    id='PandaEnv-v0',
    entry_point='Panda_Env.envs:PandaEnv',
)

register(
    id='PandaEnvPath-v0',
    entry_point='Panda_Env.envs:PandaEnvPath',
)
