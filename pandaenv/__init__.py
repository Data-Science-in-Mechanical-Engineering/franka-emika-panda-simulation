from gym.envs.registration import register

register(
    id='PandaEnvBasic-v0',
    entry_point='pandaenv.envs:PandaEnvBasic',
)

register(
    id='PandaEnvPath-v0',
    entry_point='pandaenv.envs:PandaEnvPath',
)
