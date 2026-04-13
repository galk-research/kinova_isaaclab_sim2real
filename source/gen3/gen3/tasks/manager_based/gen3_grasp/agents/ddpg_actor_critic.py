from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl.algorithms import RLAlgorithm

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.utils import get_activation


class DDPGActorCritic(ActorCritic):
    """DDPG Actor-Critic networks."""

    def __init__(self, cfg, obs_space, act_space, init_noise_std=1.0):
        super().__init__(cfg, obs_space, act_space, init_noise_std)

        # Actor network (deterministic policy)
        self.actor = nn.Sequential(
            nn.Linear(self.obs_space, cfg.hidden_dims[0]),
            get_activation(cfg.activation)(),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            get_activation(cfg.activation)(),
            nn.Linear(cfg.hidden_dims[1], self.act_space),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Critic network (Q-function)
        self.critic = nn.Sequential(
            nn.Linear(self.obs_space + self.act_space, cfg.hidden_dims[0]),
            get_activation(cfg.activation)(),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            get_activation(cfg.activation)(),
            nn.Linear(cfg.hidden_dims[1], 1)
        )

        # Initialize weights
        self.init_weights()

    def forward_actor(self, obs):
        """Forward pass through actor."""
        return self.actor(obs)

    def forward_critic(self, obs, actions):
        """Forward pass through critic."""
        x = torch.cat([obs, actions], dim=-1)
        return self.critic(x)

    def act(self, obs):
        """Sample action (deterministic for DDPG)."""
        return self.forward_actor(obs)