import torch
from .common import *
from collections import namedtuple, deque
from random import sample
from utils.hyperparams import DQN_Config as default_config

Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))

class DQN(BaseRLAlgorithm):
    def make_net(self):
        return nn.Sequential(
            nn.Linear(self.input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def act(self, state, epsilon=0):
        if np.random.random() > epsilon:
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action
    
    def update(self, S, A, R, S_new, old_net=None):
        # make mask
        done = torch.tensor([s is None for s in S_new], dtype=torch.float32).to(self.device)

        # convert S_new tuple to tensor with zeros(doesn't matter) for None
        S_new = torch.cat([
            torch.zeros(self.input_shape).to(self.device).unsqueeze(0) 
            if s is None else s for s in S_new
        ])
        
        # calculate target returns after masking
        with torch.no_grad():
            if old_net is not None:
                reward = R + self.gamma * old_net.forward(S_new).max(1)[0] * (1 - done)
            else:
                reward = R + self.gamma * self.forward(S_new).max(1)[0] * (1 - done)
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
        
    def train(self, env, episodes=1500, render=False):
        # unpack config
        epsilon = self.config.epsilon
        epsilon_min = self.config.epsilon_min
        epsilon_decay = self.config.epsilon_decay
        use_target_net = self.config.use_target_net
        replay_capacity = self.config.replay_capacity
        batch_size = self.config.batch_size
        tau = self.config.tau

        # set up
        final_log_rewards = []
        losses_log = []

        memory = deque([], maxlen=replay_capacity)
        
        if use_target_net:
            old_net = self.make_net()
            old_net.load_state_dict(self.net.state_dict())
            old_net.eval()
            old_net.requires_grad_(False)
        else:
            old_net = None

        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            log_rewards = []
            t=0
            durations_log = []
            while not done:
                if render:
                    env.render()

                # sample step from agent and env
                with torch.no_grad():
                    action = self.act(state, epsilon=epsilon)
                state_new, reward, done, _, _ = env.step(action)
                if done:
                    state_new = None
                else:
                    state_new = torch.tensor(state_new, dtype=torch.float32).to(self.device).unsqueeze(0)
                log_rewards.append(reward)
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
                loss = self.update(states, actions, rewards, new_states, old_net=old_net)
                losses_log.append(loss)
                state = state_new
                t+=1
                if use_target_net:
                    self.copy_net(old_net, tau=tau)
            final_log_rewards.append(np.sum(log_rewards))
            durations_log.append(t+1)
            if(episode % 10 == 0 and episode >= 100):
                epsilon *= epsilon_decay if epsilon > epsilon_min else 1
                
            if(episode % 100 == 0 and episode >= 100):
                print("Episode: ", episode, 
                      "Mean Reward for last 100 episodes: ", np.mean(final_log_rewards[-100:]),
                      "Mean Duration for last 100 episodes: ", np.mean(durations_log[-100:])
                      )
                print("Epsilon: ", epsilon)
        return np.array(final_log_rewards), losses_log