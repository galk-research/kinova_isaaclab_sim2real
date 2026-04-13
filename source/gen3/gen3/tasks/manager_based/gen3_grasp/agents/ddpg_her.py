from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl.runners import OffPolicyRunner

from isaaclab_rl.rsl_rl.algorithms import RLAlgorithm
from isaaclab_rl.rsl_rl.buffers import ReplayBuffer
from isaaclab_rl.rsl_rl.utils import unpad_trajectories


class DDPGHER(RLAlgorithm):
    """DDPG with Hindsight Experience Replay."""

    def __init__(self, cfg, log_dir, device):
        super().__init__(cfg, log_dir, device)

        # Actor and critic networks
        self.actor = self.cfg.actor_cls(self.cfg.actor_cfg, self.obs_space, self.act_space, self.cfg.init_noise_std)
        self.critic = self.cfg.critic_cls(self.cfg.critic_cfg, self.obs_space, self.act_space)

        # Target networks
        self.actor_target = self.cfg.actor_cls(self.cfg.actor_cfg, self.obs_space, self.act_space, self.cfg.init_noise_std)
        self.critic_target = self.cfg.critic_cls(self.cfg.critic_cfg, self.obs_space, self.act_space)

        # Copy parameters to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.learning_rate)

        # HER replay buffer
        self.storage = HERReplayBuffer(
            self.cfg.num_steps_per_env * self.cfg.num_envs,
            self.obs_space,
            self.act_space,
            self.device,
            her_strategy=self.cfg.her_strategy,
            her_k=self.cfg.her_k
        )

    def act(self, obs, critic_obs):
        """Sample actions from actor."""
        with torch.no_grad():
            actions = self.actor(obs)
        return actions

    def update(self):
        """Update networks using HER-relabeled data."""
        # Sample from HER buffer
        obs_batch, critic_obs_batch, actions_batch, rewards_batch, next_obs_batch, next_critic_obs_batch, dones_batch = self.storage.sample(self.cfg.minibatch_size)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_obs_batch)
            q_target = self.critic_target(next_critic_obs_batch, next_actions)
            q_target = rewards_batch + self.cfg.gamma * (1 - dones_batch) * q_target

        q_current = self.critic(critic_obs_batch, actions_batch)
        critic_loss = nn.MSELoss()(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        actions_pred = self.actor(obs_batch)
        actor_loss = -self.critic(critic_obs_batch, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update(self.actor, self.actor_target, self.cfg.tau)
        self._soft_update(self.critic, self.critic_target, self.cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": q_current.mean().item()
        }

    def _soft_update(self, source, target, tau):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def store_transitions(self, obs, critic_obs, actions, rewards, next_obs, next_critic_obs, dones):
        """Store transitions in HER buffer."""
        self.storage.store(obs, critic_obs, actions, rewards, next_obs, next_critic_obs, dones)


class HERReplayBuffer(ReplayBuffer):
    """Replay buffer with Hindsight Experience Replay."""

    def __init__(self, buffer_size, obs_space, act_space, device, her_strategy="future", her_k=4):
        super().__init__(buffer_size, obs_space, act_space, device)
        self.her_strategy = her_strategy
        self.her_k = her_k
        self.episode_buffer = []  # Temporary buffer for current episode

    def store(self, obs, critic_obs, actions, rewards, next_obs, next_critic_obs, dones):
        """Store transition in episode buffer."""
        transition = (obs, critic_obs, actions, rewards, next_obs, next_critic_obs, dones)
        self.episode_buffer.append(transition)

        # Check if episode ended
        if dones.any():
            self._process_episode()

    def _process_episode(self):
        """Process completed episode with HER relabeling."""
        if not self.episode_buffer:
            return

        # Store original episode
        for transition in self.episode_buffer:
            super().store(*transition)

        # Apply HER relabeling
        if self.her_strategy == "future":
            self._apply_future_her()

        self.episode_buffer = []

    def _apply_future_her(self):
        """Apply future HER strategy."""
        episode_length = len(self.episode_buffer)

        for k in range(self.her_k):
            # Sample a random future step for each transition
            for t in range(episode_length):
                # Sample future index
                future_idx = torch.randint(t, episode_length, (1,)).item()

                # Get achieved goal from future step
                _, _, _, _, next_obs_future, _, _ = self.episode_buffer[future_idx]
                new_desired_goal = self._extract_achieved_goal(next_obs_future)

                # Relabel current transition
                obs, critic_obs, actions, _, next_obs, next_critic_obs, dones = self.episode_buffer[t]

                # Update desired goal in obs
                relabeled_obs = self._relabel_obs(obs, new_desired_goal)
                relabeled_critic_obs = self._relabel_obs(critic_obs, new_desired_goal)
                relabeled_next_obs = self._relabel_obs(next_obs, new_desired_goal)
                relabeled_next_critic_obs = self._relabel_obs(next_critic_obs, new_desired_goal)

                # Recompute reward
                new_reward = self._compute_reward(relabeled_obs, actions, relabeled_next_obs)

                # Store relabeled transition
                super().store(relabeled_obs, relabeled_critic_obs, actions, new_reward,
                            relabeled_next_obs, relabeled_next_critic_obs, dones)

    def _extract_achieved_goal(self, obs):
        """Extract achieved goal from observation."""
        # Assuming achieved_goal is in obs['policy']['achieved_goal']
        return obs['policy']['achieved_goal']

    def _relabel_obs(self, obs, new_desired_goal):
        """Replace desired goal in observation."""
        relabeled_obs = obs.copy()
        relabeled_obs['policy']['desired_goal'] = new_desired_goal
        return relabeled_obs

    def _compute_reward(self, obs, actions, next_obs):
        """Recompute reward with relabeled goal."""
        achieved_goal = self._extract_achieved_goal(next_obs)
        desired_goal = obs['policy']['desired_goal']
        dist = torch.norm(achieved_goal - desired_goal, dim=-1)
        reward = -dist
        # Add success bonus
        success = (dist < 0.05).float() * 100.0
        return reward + success