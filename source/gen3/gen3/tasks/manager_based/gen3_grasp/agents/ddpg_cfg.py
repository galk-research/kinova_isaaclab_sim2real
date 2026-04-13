from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOffPolicyRunnerCfg

from .ddpg_actor_critic import DDPGActorCritic
from .ddpg_her import DDPGHER


@configclass
class DDPGActorCriticCfg:
    """DDPG Actor-Critic configuration."""
    hidden_dims = [256, 256]
    activation = "relu"
    init_noise_std = 0.1


@configclass
class DDPGAlgorithmCfg:
    """DDPG algorithm configuration."""
    actor_cls = DDPGActorCritic
    critic_cls = DDPGActorCritic
    actor_cfg = DDPGActorCriticCfg()
    critic_cfg = DDPGActorCriticCfg()
    learning_rate = 1e-4
    gamma = 0.98
    tau = 0.005
    max_grad_norm = 1.0
    her_strategy = "future"
    her_k = 4
    minibatch_size = 256


@configclass
class Gen3GraspDDPGRunnerCfg(RslRlOffPolicyRunnerCfg):
    """DDPG runner configuration for grasp task."""
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "grasp_gen3_ddpg_her"
    run_name = ""
    resume = False
    clip_actions = 1.0
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    policy = DDPGActorCriticCfg(
        hidden_dims=[256, 256],
        activation="relu",
        init_noise_std=0.1
    )
    algorithm = DDPGAlgorithmCfg(
        learning_rate=1e-4,
        gamma=0.98,
        tau=0.005,
        her_strategy="future",
        her_k=4,
        minibatch_size=256
    )

    def __post_init__(self):
        # Use our custom DDPG algorithm
        self.algorithm_cls = DDPGHER