from .make_model import make_model
from .make_env import make_env
from .plotter import plot_rewards
import torch
import numpy as np
import random
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(env, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)