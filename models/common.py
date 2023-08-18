import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
import numpy as np
from tqdm import tqdm
from utils.hyperparams import Base_Config
class BaseRLAlgorithm(nn.Module, abc.ABC):
    def __init__(self, input_shape, num_actions, config=Base_Config()):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.config = config
        self.gamma = config.gamma
        if len(self.input_shape) > 1:
            self.transpose = True
        else:
            self.transpose = False
        self.net = self.make_net()
        self._optimizer = None # lazy init
        self._device = None # lazy init

    def forward(self, state):
        if self.transpose:
            state = state.transpose(1, 3).transpose(2, 3)
        return self.net(state)

    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def train(self, env, num_episodes=1000, max_steps=1000, render=False, verbose=False):
        raise NotImplementedError

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self.config.optimizer(self.parameters(), lr=self.config.lr)
        return self._optimizer
    
    @property
    def device(self):
        if self._device is None:
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                print("No parameters found, using cpu")
                self._device = torch.device("cpu")
        return self._device
    
    def make_net(self):
        if self.transpose:
            # hxwxc = 210x160x3 for atari breakout
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=8, stride=2), # 210x160x3 -> 102x77x16
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # 102x77x16 -> 51x38x16
                nn.Dropout(p=0.2),
                nn.Conv2d(16, 32, kernel_size=4, stride=2), # 51x38x16 -> 24x18x32
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # 24x18x32 -> 12x9x32
                # nn.Dropout(p=0.4),
                nn.Conv2d(32, 64, kernel_size=3, stride=1), # 12x9x32 -> 10x7x64
                nn.ReLU(),
                nn.Flatten(), # 10x7x64 -> 4480
                # nn.Dropout(p=0.4),
                nn.Linear(4480, 256),
                nn.ReLU(),
                # nn.Dropout(p=0.6),
                nn.Linear(256, self.num_actions)
            )
        else:
            return nn.Sequential(
                nn.Linear(self.input_shape[0], self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=0.6),
                nn.Linear(self.config.hidden_size, self.num_actions)
            )