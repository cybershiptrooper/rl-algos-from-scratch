from models import *
from utils.hyperparams import *

model_map = {
    "dqn" : DQN,
    "reinforce" : Reinforce,
    # "a2c" : A2C,
    # "ppo" : PPO,
    # "ddpg" : DDPG,
}

config_map = {
    "reinforce" : Base_Config,
    "dqn" : DQN_Config,
    # "a2c" : A2C_Config,
    # "ppo" : PPO_Config,
    # "ddpg" : DDPG_Config,
}

def make_model(model_name, env, device="cpu", **kwargs):
    config = config_map[model_name]()
    return model_map[model_name](env.observation_space.shape, env.action_space.n, config=config, **kwargs).to(device)