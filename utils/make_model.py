from models import *
from utils.hyperparams import *

model_map = {
    "dqn" : DQN,
    "reinforce" : Reinforce,
    "a2c" : A2C,
    # "ppo" : PPO,
    "ddpg" : DDPG,
}

config_map = {
    "reinforce" : Base_Config,
    "dqn" : DQN_Config,
    "a2c" : A2C_Config,
    # "ppo" : PPO_Config,
    "ddpg" : DDPG_Config,
}

def make_model(model_name, env, device="cpu", **kwargs):
    config = config_map[model_name]()
    if model_name == "ddpg": # continous action space
        config.action_min = torch.tensor(env.action_space.low).to(device)
        config.action_max = torch.tensor(env.action_space.high).to(device)
        actions = env.action_space.shape[0]
    else:
        actions = env.action_space.n

    return model_map[model_name](env.observation_space.shape, actions, config=config, **kwargs).to(device)