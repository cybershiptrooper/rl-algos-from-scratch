from dataclasses import dataclass, MISSING
import torch

@dataclass
class Base_Config:
    gamma: float = 0.99
    lr: float = 1e-3
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    hidden_size: int = 50

@dataclass
class DQN_Config(Base_Config):
    epsilon: float = 0.95
    epsilon_min: float = 0.002
    epsilon_decay: float = .9999
    replay_capacity: int = 10000
    batch_size: int = 128
    tau: float = 0.005
    change_target_net_every: int = 1
    optimizer: torch.optim.Optimizer = torch.optim.AdamW
    lr: float = 1e-4
    hidden_size: int = 128

@dataclass
class A2C_Config(Base_Config):
    actor_lr: float = 1e-2
    critic_lr: float = 1e-2
    actor_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    critic_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    gamma: float = 0.99
    hidden_size: int = 16
    mc: bool = True

@dataclass
class TD0_A2C_Config(Base_Config):
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    actor_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    critic_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    gamma: float = 0.99
    hidden_size: int = 128
    mc: bool = True

@dataclass
class DDPG_Config(Base_Config):
    epsilon: float = 0.95
    epsilon_min: float = 0.05
    epsilon_decay: float = .9999
    noise_var: float = 0.1
    replay_capacity: int = 20000
    batch_size: int = 128
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    actor_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    critic_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    hidden_size: int = 128
    action_min: torch.tensor = torch.tensor([-1.0])
    action_max: torch.tensor = torch.tensor([1.0])

    test_every: int = 10
    test_episodes: int = 10