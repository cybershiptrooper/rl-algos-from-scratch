import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, running_avg=20):
    # rewards_mean = rewards.reshape(-1, running_avg).mean(axis=1) # reshape ravels for each 'average' elements
    rewards_running_avg = np.convolve(rewards, np.ones(running_avg)/running_avg, mode='valid')
    rewards_running_avg = np.concatenate((np.zeros(running_avg-1), rewards_running_avg))
    plt.plot(rewards, color='green')
    plt.plot(rewards_running_avg, color='red')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.show()