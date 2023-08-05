from dataclasses import dataclass, MISSING
import torch

@dataclass
class Base_Config:
    gamma: float = 0.99
    lr: float = 1e-3
    optimizer: torch.optim.Optimizer = torch.optim.Adam

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