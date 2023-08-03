import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc

class BaseRLAlgorithm(nn.Module, abc.ABC):
    def __init__(self, input_shape, num_actions, optimizer, gamma=0.99):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.optimizer = optimizer

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")