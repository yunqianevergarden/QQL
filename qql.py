"""
QQL (Quantile Q-Learning) implementation

This implementation is based on the IQL implementation from the CORL (Clean Offline Reinforcement Learning) repository
but modified to implement QQL methodology:
- Uses quantile regression for value function learning instead of asymmetric L2 loss
- Implements gap correction in Q-function updates
- Employs dual value functions with different quantile parameters
- Uses imagination-based value function updates for better exploration
- Combines advantages from both value functions for policy learning

CORL Repo: https://github.com/tinkoff-ai/CORL
"""

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from math import exp

TensorBatch = List[torch.Tensor]

# Constants
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
gamma = 0.5772155649  # Euler-Mascheroni constant

@dataclass
class TrainConfig:
    """Configuration class for training parameters"""
    project: str = "QQL"
    group: str = "QQL-VR"
    name: str = "QQL-VR"
    env: str = "hopper-medium-expert-v2"
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 0.1
    iql_tau_soft: float = 1 - exp(-1)
    gamma = 0.5772155649
    iql_tau: float = 1 - exp(-exp(gamma))
    iql_tau_low: float = 1 - exp(-exp(-gamma))
    iql_deterministic: bool = False
    max_timesteps: int = int(1e6)
    buffer_size: int = 2_000_000
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    bc_ratio: float = 1.0
    mild: float = 1.0
    actor_dropout: Optional[float] = None
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self):
        """Generate unique run name after initialization"""
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Perform soft update of target network parameters
    
    Args:
        target (nn.Module): Target network to be updated
        source (nn.Module): Source network providing the parameters
        tau (float): Soft update coefficient (0 < tau < 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation of states
    
    Args:
        states (np.ndarray): Input states array of shape (N, state_dim)
        eps (float): Small constant to add to standard deviation for numerical stability
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and standard deviation arrays of shape (state_dim,)
    """
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize states using mean and standard deviation
    
    Args:
        states (np.ndarray): Input states array of shape (N, state_dim)
        mean (np.ndarray): Mean values for normalization of shape (state_dim,)
        std (np.ndarray): Standard deviation values for normalization of shape (state_dim,)
        
    Returns:
        np.ndarray: Normalized states array of shape (N, state_dim)
    """
    return (states - mean) / std

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    """Wrap environment with state normalization and reward scaling
    
    Args:
        env (gym.Env): Original gym environment
        state_mean (Union[np.ndarray, float]): Mean values for state normalization
        state_std (Union[np.ndarray, float]): Standard deviation values for state normalization
        reward_scale (float): Scaling factor for rewards
        
    Returns:
        gym.Env: Wrapped environment with state normalization and reward scaling
    """
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

class ReplayBuffer:
    """Replay buffer for storing and sampling transitions"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        """Initialize replay buffer with specified dimensions and size"""
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor
        
        Args:
            data (np.ndarray): Input numpy array
            
        Returns:
            torch.Tensor: Tensor with float32 dtype on the specified device
        """
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """Load d4rl format dataset into replay buffer
        
        Args:
            data (Dict[str, np.ndarray]): Dictionary containing d4rl dataset with keys:
                - 'observations': State observations array
                - 'actions': Action array
                - 'rewards': Reward array
                - 'next_observations': Next state observations array
                - 'terminals': Terminal flags array
                
        Raises:
            ValueError: If replay buffer is not empty or dataset is too large
        """
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        """Sample a batch of transitions from replay buffer
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            TensorBatch: List of tensors [states, actions, rewards, next_states, dones]
                - states: Tensor of shape (batch_size, state_dim)
                - actions: Tensor of shape (batch_size, action_dim)
                - rewards: Tensor of shape (batch_size, 1)
                - next_states: Tensor of shape (batch_size, state_dim)
                - dones: Tensor of shape (batch_size, 1)
        """
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        """Add new transition to replay buffer (not implemented for offline RL)"""
        raise NotImplementedError

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    """Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
        env (Optional[gym.Env]): Gym environment to set seed for
        deterministic_torch (bool): Whether to use deterministic PyTorch algorithms
    """
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    """Initialize wandb for experiment tracking
    
    Args:
        config (dict): Configuration dictionary containing project, group, and name keys
    """
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

    if not os.path.exists("wandb_offline"):
        os.makedirs("wandb_offline")
    
    print("Running in offline mode. Data will be saved locally in ./wandb directory.")
    print("To sync after reconnecting, run: wandb sync wandb/offline-run-*")

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    """Evaluate actor policy over multiple episodes
    
    Args:
        env (gym.Env): Gym environment for evaluation
        actor (nn.Module): Actor policy network
        device (str): Device to run evaluation on ('cpu' or 'cuda')
        n_episodes (int): Number of episodes to evaluate
        seed (int): Random seed for environment
        
    Returns:
        np.ndarray: Array of episode rewards of shape (n_episodes,)
    """
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

def return_reward_range(dataset, max_episode_steps):
    """Calculate reward range for normalization
    
    Args:
        dataset (dict): Dataset dictionary containing 'rewards' and 'terminals' keys
        max_episode_steps (int): Maximum number of steps per episode
        
    Returns:
        Tuple[float, float]: Minimum and maximum episode returns
    """
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(dataset, env_name, max_episode_steps=1000):
    """Modify rewards for specific environments
    
    Args:
        dataset (dict): Dataset dictionary containing 'rewards' key
        env_name (str): Name of the environment
        max_episode_steps (int, optional): Maximum episode steps. Defaults to 1000.
    """
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

def quantile_loss(u, quantile, weight=1.0):
    """Compute quantile loss for quantile regression
    
    Args:
        u (torch.Tensor): Difference between real value and predicted value (real_value - predict_value)
        quantile (float): Quantile parameter (0 < quantile < 1)
        weight (float, optional): Weight for the loss. Defaults to 1.0.
        
    Returns:
        torch.Tensor: Quantile loss value
    """
    # u = real_value - predict_value
    loss = torch.mean(torch.max((quantile - 1) * u, quantile * u) * weight)
    return loss

class Squeeze(nn.Module):
    """Squeeze layer for removing dimensions"""
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    """Multi-layer perceptron network"""
    
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GaussianPolicy(nn.Module):
    """Gaussian policy network for continuous action spaces"""
    
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """Generate action for given state
        
        Args:
            state (np.ndarray): Current state observation
            device (str, optional): Device to run computation on. Defaults to "cpu".
            
        Returns:
            np.ndarray: Action array of shape (action_dim,)
        """
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

class DeterministicPolicy(nn.Module):
    """Deterministic policy network for continuous action spaces"""
    
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """Generate deterministic action for given state
        
        Args:
            state (np.ndarray): Current state observation
            device (str, optional): Device to run computation on. Defaults to "cpu".
            
        Returns:
            np.ndarray: Deterministic action array of shape (action_dim,)
        """
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )

class TwinQ(nn.Module):
    """Twin Q-network for reducing overestimation bias"""
    
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both Q-networks
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q-values from Q1 and Q2 networks
        """
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning minimum of both Q-values
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim)
            
        Returns:
            torch.Tensor: Minimum Q-value from both networks
        """
        return torch.min(*self.both(state, action))



class ValueFunction(nn.Module):
    """Value function network"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)



class ImplicitQLearning:
    """Quantile Q-Learning (QQL) agent implementation
    
    Note: While this class is named 'ImplicitQLearning' for historical reasons, 
    it implements QQL (Quantile Q-Learning) which is fundamentally different from IQL:
    
    Key differences from IQL:
    1. Uses quantile regression loss instead of asymmetric L2 loss for value function learning
    2. Implements gap correction in Q-function updates
    3. Employs dual value functions with different quantile parameters
    4. Uses imagination-based value function updates for better exploration
    5. Combines advantages from both value functions for policy learning
    
    The name 'ImplicitQLearning' is kept for compatibility with existing code structure,
    but the implementation follows QQL methodology.
    """
    
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        v_soft_network: nn.Module,
        v_soft_optimizer: torch.optim.Optimizer,
        iql_tau_soft: float = 1 - exp(-1),
        iql_tau: float = 1 - exp(-exp(gamma)),
        iql_tau_low: float = 1 - exp(-exp(-gamma)),
        beta: float = 1.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        bc_ratio: float = 1.0,
        mild: float = 0.25,
        device: str = "cpu",
    ):
        """Initialize QQL agent with all networks and optimizers"""
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.v_soft_f = v_soft_network
        self.vf = v_network
        self.actor = actor
        self.v_soft_optimizer = v_soft_optimizer
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau_low = iql_tau_low
        self.iql_tau_soft = iql_tau_soft
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.bc_ratio = bc_ratio
        self.mild = mild
        self.total_it = 0
        self.device = device

    def _update_v_soft(self, observations, actions, log_dict) -> torch.Tensor:
        """Update soft value function using quantile loss
        Corresponds to the $V(s)$ update in the QQL paper
        Args:
            observations (torch.Tensor): Batch of observations
            actions (torch.Tensor): Batch of actions
            log_dict (Dict): Dictionary to log training metrics
            
        Returns:
            torch.Tensor: Soft advantage values
        """
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.v_soft_f(observations)
        adv_soft = target_q - v
        v_soft_loss = quantile_loss(adv_soft, self.iql_tau_soft)
        log_dict["value_soft_loss"] = v_soft_loss.item()
        self.v_soft_optimizer.zero_grad()
        v_soft_loss.backward()
        self.v_soft_optimizer.step()
        return adv_soft

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        """Update value function using quantile loss
        Corresponds to the $\hat{V}(s)$ update in the QQL paper
        
        Args:
            observations (torch.Tensor): Batch of observations
            actions (torch.Tensor): Batch of actions
            log_dict (Dict): Dictionary to log training metrics
            
        Returns:
            torch.Tensor: Advantage values
        """
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = quantile_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        """Update Q-functions with gap correction
        
        Args:
            next_v (torch.Tensor): Next state values
            observations (torch.Tensor): Current state observations
            actions (torch.Tensor): Actions taken
            rewards (torch.Tensor): Rewards received
            terminals (torch.Tensor): Terminal flags
            log_dict (Dict): Dictionary to log training metrics
        """
        gap = (self.vf(observations) - self.v_soft_f(observations)).detach()
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach() - gap
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        
        q1, q2 = qs
        log_dict["q1_mean"] = q1.mean().item()
        log_dict["q2_mean"] = q2.mean().item()
        log_dict["q_mean"] = ((q1 + q2) / 2).mean().item()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)



    def _action(self, observations: torch.Tensor):
        """Generate actions using current policy
        
        Args:
            observations (torch.Tensor): State observations
            
        Returns:
            torch.Tensor: Generated actions
        """
        with torch.no_grad():
            observations = torch.tensor(observations, device=self.device, dtype=torch.float32)
            dist = self.actor(observations)
            act_ = dist.sample()
            act_ = torch.clamp(self.max_action * act_, -self.max_action, self.max_action)
        return act_

    def _update_v_soft_imagination(self, observations: torch.Tensor, log_dict: Dict):
        """Update soft value function using imagined actions
        
        Args:
            observations (torch.Tensor): State observations
            log_dict (Dict): Dictionary to log training metrics
            
        Returns:
            torch.Tensor: Soft advantage values from imagined actions
        """
        with torch.no_grad():
            actions = self._action(observations)
            target_q = self.q_target(observations, actions)
            
        v = self.v_soft_f(observations)
        adv_soft = target_q - v
        # Conservative estimation: use iql_tau_low for more conservative value estimates
        # To disable conservative estimation (QQL w/o CE), change to:
        # v_soft_loss = quantile_loss(adv_soft, self.iql_tau_soft, self.mild)
        v_soft_loss = quantile_loss(adv_soft, self.iql_tau_low, self.mild)
        log_dict["value_soft_loss_imagination_update"] = v_soft_loss.item()
        self.v_soft_optimizer.zero_grad()
        v_soft_loss.backward()
        self.v_soft_optimizer.step()
        return adv_soft
    
    def _update_v_imagination(self, observations: torch.Tensor, log_dict: Dict):
        """Update value function using imagined actions
        
        Args:
            observations (torch.Tensor): State observations
            log_dict (Dict): Dictionary to log training metrics
            
        Returns:
            torch.Tensor: Advantage values from imagined actions
        """
        with torch.no_grad():
            actions = self._action(observations)
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
         # Conservative estimation: use iql_tau_soft for more conservative value estimates
        # To disable conservative estimation (QQL w/o CE), change to:
        # v_loss = quantile_loss(adv_soft, self.iql_tau, self.mild)
        v_loss = quantile_loss(adv, self.iql_tau_soft, self.mild)
        log_dict["value_loss_imagination_update"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_policy_mild_extrapolate(self, adv: torch.Tensor, adv_soft: torch.Tensor, observations: torch.Tensor, actions: torch.Tensor, log_dict: Dict):
        """Update policy using combined advantages from both value functions
        
        This is the core policy learning component of QQL:
        - Combines advantages from both optimal and soft value functions
        - Implements adaptive temperature scaling based on value function gap
        - Uses mild extrapolation to improve policy generalization beyond dataset
        - Balances between behavior cloning and advantage-weighted learning
        
        Args:
            adv (torch.Tensor): Advantage values from optimal value function (V_hat)
            adv_soft (torch.Tensor): Advantage values from soft value function (V_soft)
            observations (torch.Tensor): State observations
            actions (torch.Tensor): Actions from dataset
            log_dict (Dict): Dictionary to log training metrics
        """
        with torch.no_grad():
            beta_ = self.beta + abs(self.vf(observations) - self.v_soft_f(observations)) / gamma
            alpha = 1 / beta_
        adv_combine = adv / self.bc_ratio + adv_soft
        exp_adv_combine = torch.exp(alpha * (adv_combine.detach())).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv_combine * bc_losses)
        log_dict["actor_loss_extrapolate"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log_dict["beta_"] = beta_

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """Perform one training step
        
        Args:
            batch (TensorBatch): Batch of transitions [observations, actions, rewards, next_observations, dones]
            
        Returns:
            Dict[str, float]: Dictionary containing training metrics and losses
            
        Note: To use QQL without Value Regularization (VR), comment out the imagination-based updates:
            # self._update_v_imagination(next_observations, log_dict)
            # self._update_v_soft_imagination(next_observations, log_dict)
        """
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        
        # Update value functions
        adv_soft = self._update_v_soft(observations, actions, log_dict)
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        
        # Update Q-functions
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        
        # Update policy using combined advantages from both value functions
        self._update_policy_mild_extrapolate(adv, adv_soft, observations, actions, log_dict)

        # Update value functions using imagination (Value Regulation)
        self._update_v_imagination(next_observations, log_dict)
        self._update_v_soft_imagination(next_observations, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving
        
        Returns:
            Dict[str, Any]: Dictionary containing all network states and training progress
        """
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_soft_f": self.v_soft_f.state_dict(),
            "v_soft_optimizer": self.v_soft_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary
        
        Args:
            state_dict (Dict[str, Any]): Dictionary containing saved network states and training progress
        """
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.v_soft_f.load_state_dict(state_dict["v_soft_f"])
        self.v_soft_optimizer.load_state_dict(state_dict["v_soft_optimizer"])

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

@pyrallis.wrap()
def train(config: TrainConfig):
    """Main training function for QQL agent
    
    This function demonstrates a complete training pipeline for QQL:
    1. Environment and dataset setup
    2. Data preprocessing and normalization
    3. Network initialization and configuration
    4. Training loop with evaluation
    5. Model checkpointing and logging
    
    Args:
        config (TrainConfig): Training configuration containing all hyperparameters
    """
    # Initialize environment and get dimensions
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load offline dataset from D4RL
    dataset = d4rl.qlearning_dataset(env)

    # Data preprocessing: reward normalization for specific environments
    if config.normalize_reward:
        modify_reward(dataset, config.env)

    # State normalization: compute statistics and normalize states
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    # Apply state normalization to both current and next observations
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    # Wrap environment with normalization for evaluation
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    # Initialize replay buffer and load dataset
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    # Get action bounds for policy networks
    max_action = float(env.action_space.high[0])

    # Setup checkpointing directory and save configuration
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set random seeds for reproducibility
    seed = config.seed
    set_seed(seed, env)

    # Initialize QQL networks: Q-functions, value functions, and policy
    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_soft_network = ValueFunction(state_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)

    # Initialize optimizers for all networks
    v_soft_optimizer = torch.optim.Adam(v_soft_network.parameters(), lr=config.vf_lr)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    # Prepare configuration dictionary for QQL trainer
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_soft_network": v_soft_network,
        "v_soft_optimizer": v_soft_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "beta": config.beta,
        "iql_tau_low": config.iql_tau_low,
        "iql_tau_soft": config.iql_tau_soft,
        "iql_tau": config.iql_tau,
        "bc_ratio": config.bc_ratio,
        "max_steps": config.max_timesteps,
        "mild": config.mild,
    }

    # Print training information
    print("---------------------------------------")
    print(f"Training QQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize QQL trainer with all components
    trainer = ImplicitQLearning(**kwargs)

    # Load pre-trained model if specified
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # Initialize experiment tracking with wandb
    wandb_init(asdict(config))

    # Main training loop
    evaluations = []
    for t in range(int(config.max_timesteps)):
        # Sample batch from replay buffer and move to device
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        # Perform one training step and log metrics
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        
        # Periodic evaluation and checkpointing
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            # Evaluate current policy performance
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            # Save model checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            # Log evaluation results
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it
            )

if __name__ == "__main__":
    # Start training when script is run directly
    train() 