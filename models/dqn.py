import torch
from .common import *
from collections import namedtuple, deque
from random import sample
from utils.hyperparams import DQN_Config as default_config

Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))

class DQN(BaseRLAlgorithm):
    def act(self, state, epsilon=0):
        if np.random.random() > epsilon:
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action
    
    def update(self, S, A, R, S_new, target_net):
        # make mask
        done = torch.tensor([s is None for s in S_new])
        S_new = torch.cat([s for s in S_new if s is not None])
        
        # calculate target returns after masking
        with torch.no_grad():
            reward = R
            reward[done==0] += self.gamma * target_net.forward(S_new).max(1)[0]
        # update network
        criterion = nn.SmoothL1Loss()
        loss = criterion(reward, self.forward(S)[torch.arange(len(A)), A])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.optimizer.step()
        return loss.item()
    
    def copy_net(self, net, tau=0.005):
        for param, target_param in zip(self.net.parameters(), net.parameters()): # soft-update
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        net.eval()
        net.requires_grad_(False)
        
    def train(self, env, episodes=800, render=False):
        # unpack config
        epsilon = self.config.epsilon
        epsilon_max = self.config.epsilon
        epsilon_min = self.config.epsilon_min
        epsilon_decay = self.config.epsilon_decay
        replay_capacity = self.config.replay_capacity
        batch_size = self.config.batch_size
        tau = self.config.tau
        set_target_net_every = self.config.change_target_net_every

        # set up
        epi_rewards_logger = []
        epi_losses_logger = []

        memory = deque([], maxlen=replay_capacity)
        
        target_net = self.make_net().to(self.device)
        target_net.load_state_dict(self.net.state_dict())
        target_net.eval()
        target_net.requires_grad_(False)
        steps = 0
        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            rewards_logger = []
            losses_logger = []
            while not done:
                if render:
                    env.render()

                # sample step from agent and env
                with torch.no_grad():
                    action = self.act(state, epsilon=epsilon)
                state_new, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
                if done:
                    state_new = None
                else:
                    state_new = torch.tensor(state_new, dtype=torch.float32).to(self.device).unsqueeze(0)
                rewards_logger.append(reward)
                reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = torch.tensor(action, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # update and sample memory
                memory.append(Transition(state, action, state_new, reward))
                if len(memory) < batch_size:
                    continue
                batch = sample(memory, batch_size)
                batch = Transition(*zip(*batch))
                states = torch.cat(batch.state)
                actions = torch.cat(batch.action)
                rewards = torch.cat(batch.reward)
                new_states = batch.new_state
                
                # update network
                loss = self.update(states, actions, rewards, new_states, target_net=target_net)
                losses_logger.append(loss)
                state = state_new
                if(steps % set_target_net_every == 0):
                    self.copy_net(target_net, tau=tau)
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            epi_rewards_logger.append(np.sum(rewards_logger))
            # log
            if len(losses_logger) == 0:
                epi_losses_logger.append(-1)
            else:
                epi_losses_logger.append(np.mean(losses_logger))
                
            if(episode % 100 == 0 and episode >= 100):
                print("Episode: ", episode, 
                      "\nMean Reward for last 100 episodes: ", np.mean(epi_rewards_logger[-100:]),
                      "\nEpsilon: ", epsilon)
                print("steps: ", steps)
        return np.array(epi_rewards_logger), np.array(epi_losses_logger)