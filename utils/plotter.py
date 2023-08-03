import matplotlib.pyplot as plt

def plot_rewards(rewards, running_avg=10):
    rewards = rewards.reshape(-1, running_avg).mean(axis=1) # reshape ravels for each 'average' elements
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.show()