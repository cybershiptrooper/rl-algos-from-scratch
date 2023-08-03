import argparse
from utils import make_model, make_env, plot_rewards, device
import numpy as np
from tqdm import tqdm
def run_model(env, model, episodes=1500, render=False):
    env.reset()
    final_rewards = []
    for episode in tqdm(range(episodes)):
        done = False
        state, _ = env.reset()
        rewards = []
        log_probs = []
        # sample an episode
        while not done:
            if render:
                env.render()
            action, log_prob = model.act(state)
            log_probs.append(log_prob)
            # state, reward, terminated, truncated, info 
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
        env.reset()
        rewards = np.array(rewards)
        # update the model
        loss = model.update(rewards, log_probs)
        # print(f"Episode: {episode}, Total Reward: {rewards.sum()}, Loss: {loss}")
        final_rewards.append(rewards.sum())
    return np.array(final_rewards)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="cartpole")
    argparser.add_argument("--model", type=str, default="reinforce")
    args = argparser.parse_args()

    env = make_env(args.env)
    import torch
    optimizer = torch.optim.Adam
    model = make_model(args.model, env, optimizer=optimizer)
    rewards = run_model(env, model)
    plot_rewards(rewards)
    env.close()