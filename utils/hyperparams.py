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
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.90
    use_target_net: bool = True
    replay_capacity: int = 20000
    batch_size: int = 1024
    tau: float = 0.001
