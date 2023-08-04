import argparse
from utils import make_model, make_env, plot_rewards, seed_everything, device

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="cartpole")
    argparser.add_argument("--model", type=str, default="reinforce")
    args = argparser.parse_args()

    env = make_env(args.env)
    seed_everything(env)
    model = make_model(args.model, env)
    rewards, losses = model.train(env)
    plot_rewards(rewards)
    env.close()