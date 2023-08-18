from collections import namedtuple, deque
from random import sample
from tqdm import tqdm
import torch
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, state_new, reward):
        self.memory.append(Transition(state, action, state_new, reward))
    
    def sample_batch(self, batch_size):
        if batch_size > len(self.memory):
            raise ValueError("batch_size must be less than or equal to the size of the replay buffer")
        batch = sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        new_states = batch.new_state
        return states, actions, new_states, rewards
    
    def __len__(self):
        return len(self.memory)

def soft_update(from_net, target_net, tau=0.005):
    for target_param, param in zip(target_net.parameters(), from_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def test(agent, env, episodes=100, render=False):
        epi_rewards_logger = []
        # agent.net.actor.eval()
        # net.critic.eval()
        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            rewards_logger = []
            while not done:
                if render:
                    env.render()
                
                # sample step from environment and agent
                with torch.no_grad():
                    action = agent.act(state)
                action_scalar = action.squeeze(0).cpu().numpy()
                state_new, reward, terminated, truncated, _ = env.step(action_scalar)
                done = terminated or truncated
                if done:
                    state_new = None
                else:
                    state_new = torch.tensor(state_new, dtype=torch.float32).unsqueeze(0).to(agent.device)
                state = state_new
                # print(reward, action_scalar, state_new, "terminated" if terminated else "truncated" if truncated else "running")
                rewards_logger.append(reward)
            epi_rewards_logger.append(np.sum(rewards_logger))
        # self.net.actor.train()
        # self.net.critic.train()
        return epi_rewards_logger