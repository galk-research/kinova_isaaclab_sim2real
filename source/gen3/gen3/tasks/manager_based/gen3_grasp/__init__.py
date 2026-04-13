import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Gen3-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Gen3GraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ddpg_cfg:Gen3GraspDDPGRunnerCfg",
    },
)
