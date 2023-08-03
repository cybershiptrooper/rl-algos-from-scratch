from .make_model import make_model
from .make_env import make_env
from .plotter import plot_rewards
import torch

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"