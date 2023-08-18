import gymnasium as gym

envmap = {
    "cartpole" : "CartPole-v1",
    "pendulum" : "Pendulum-v0",
    "atari" : "ALE/Breakout-v5",
    "mountaincar" : "MountainCarContinuous-v0",
}

def make_env(env_name, seed=-1, render_mode=False, load_model=False):
    env = gym.make(envmap[env_name], render_mode="rgb_array" if render_mode else None)
    if seed >= 0:
        env.seed(seed)
    return env