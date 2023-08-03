import argparse
from utils import make_model, make_env, plot_rewards, device


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="cartpole")
    argparser.add_argument("--model", type=str, default="reinforce")
    args = argparser.parse_args()

    env = make_env(args.env)
    import torch
    # torch.manual_seed(0)
    # np.random.seed(0)
    optimizer = torch.optim.Adam
    model = make_model(args.model, env, optimizer=optimizer)
    rewards, losses = model.train(env)
    plot_rewards(rewards)
    env.close()