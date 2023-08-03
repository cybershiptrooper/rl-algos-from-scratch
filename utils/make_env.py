import gymnasium as gym

envmap = {
    "cartpole" : "CartPole-v1",
    "pendulum" : "Pendulum-v0",
    "atari" : "Breakout-v0",
}

def make_env(env_name, seed=-1, render_mode=False, load_model=False):
    env = gym.make(envmap[env_name])
    if seed >= 0:
        env.seed(seed)
    if render_mode:
        env.render("human")
    return env