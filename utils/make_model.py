from models import *

model_map = {
    "dqn" : DQN,
    "reinforce" : Reinforce,
    # "a2c" : A2C,
    # "ppo" : PPO,
    # "ddpg" : DDPG,
}

def make_model(model_name, env, device="cpu", **kwargs):
    return model_map[model_name](env.observation_space.shape, env.action_space.n, **kwargs).to(device)