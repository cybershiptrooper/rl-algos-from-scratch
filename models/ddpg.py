from .common import *
from collections import namedtuple, deque
from random import sample

ac_pair = namedtuple('ac_pair', ('actor', 'critic'))
Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))

class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        # self.min = min
        # self.max = max
        self.mean = (min + max) / 2
        self.scale = (max - min) / 2
    def forward(self, x):
        # return torch.clamp(x.unsqueeze(-1),
        #                      min=self.min.unsqueeze(-1), 
        #                      max=self.max.unsqueeze(-1)).squeeze(-1) 
        return torch.tanh(x) * self.scale + self.mean
class DDPG(BaseRLAlgorithm):

    def make_net(self):
        if self.transpose:
            raise NotImplementedError
        else:
            # num actions is the dimension of the action space here
            actor = nn.Sequential(
                    nn.Linear(self.input_shape[0], self.config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                    nn.ReLU(),
                    # nn.Dropout(p=0.6),
                    nn.Linear(self.config.hidden_size, self.num_actions),
                    Clamp(self.config.action_min, self.config.action_max)
                    # nn.Softmax(dim=-1)
            )

            critic = nn.Sequential(
                nn.Linear(self.input_shape[0] + self.num_actions, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=0.6),
                nn.Linear(self.config.hidden_size, 1)
            )
            return ac_pair(actor, critic)
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            actor_optim = self.config.actor_optimizer(self.net.actor.parameters(), lr = self.config.actor_lr)
            critic_optim = self.config.critic_optimizer(self.net.critic.parameters(), lr = self.config.critic_lr)
            self._optimizer =  ac_pair(actor_optim, critic_optim)
        return self._optimizer
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def copy_net(self, net, tau=0.005):
        for param, target_param in zip(self.net.actor.parameters(), net.actor.parameters()): # soft-update actor
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.net.critic.parameters(), net.critic.parameters()): # soft-update critic
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        net.actor.eval()
        net.critic.eval()
        net.actor.requires_grad_(False)
        net.critic.requires_grad_(False)

    def act(self, state, noise_scale=1.0
            ):
        # gaussian noise
        noise = torch.randn(self.num_actions)*(self.config.noise_var**0.5)
        action = self.net.actor(state)
        action = (action + noise_scale * noise)
        # clamp actions- I've done it like this to make it work when 
        # action clamps vary across dimensions
        action = torch.clamp(action,
                             min=self.config.action_min.unsqueeze(-1), 
                             max=self.config.action_max.unsqueeze(-1))
        return action
    
    def update(self, S, A, R, S_new, target_net):
        done = torch.tensor([s is None for s in S_new])
        S_new = torch.cat([s for s in S_new if s is not None])

        # calculate target returns after masking
        with torch.no_grad():
            reward = R
            critic_input = torch.cat([S_new, target_net.actor(S_new)], dim=-1)
            reward[done==0] += self.gamma * target_net.critic(critic_input).squeeze(-1)
        critic_loss = F.mse_loss(
            self.net.critic(torch.cat([S, A], dim=-1)), 
            reward.unsqueeze(-1)
        )
        actor_out =  self.net.actor(S)
        actor_loss = -self.net.critic(
            torch.cat([S,actor_out], dim=-1)
        ).mean()
        # print(actor_loss, critic_loss)
        self.optimizer.actor.zero_grad()
        self.optimizer.critic.zero_grad()

        actor_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.actor.parameters(), 10)
        torch.nn.utils.clip_grad_value_(self.net.actor.parameters(), 10)
        self.optimizer.actor.step()
        self.optimizer.critic.step()

        # soft update target net
        self.copy_net(target_net, tau=self.config.tau)

        return [actor_loss.item(), critic_loss.item()]
    
    def test(self, env, episodes=100, render=False):
        epi_rewards_logger = []
        self.net.actor.eval()
        self.net.critic.eval()
        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            rewards_logger = []
            while not done:
                if render:
                    env.render()
                
                # sample step from environment and agent
                with torch.no_grad():
                    action = self.act(state, noise_scale=0.0)
                action_scalar = action.squeeze(0).cpu().numpy()
                state_new, reward, terminated, truncated, _ = env.step(action_scalar)
                done = terminated or truncated
                if done:
                    state_new = None
                else:
                    state_new = torch.tensor(state_new, dtype=torch.float32).unsqueeze(0).to(self.device)
                state = state_new
                print(reward, action_scalar, state_new, "terminated" if terminated else "truncated" if truncated else "running")
                rewards_logger.append(reward)
            epi_rewards_logger.append(np.sum(rewards_logger))
        self.net.actor.train()
        self.net.critic.train()
        return epi_rewards_logger

    def train(self, env, episodes=1000, render=False):
        if env.action_space.__class__.__name__ == 'Discrete':
            raise ValueError('DDPG is only for continuous action spaces')
        
        # unpack config
        epsilon = self.config.epsilon
        epsilon_max = self.config.epsilon
        epsilon_min = self.config.epsilon_min
        epsilon_decay = self.config.epsilon_decay
        replay_capacity = self.config.replay_capacity
        batch_size = self.config.batch_size
        test_every = self.config.test_every
        test_episodes = self.config.test_episodes

        # set up
        epi_rewards_logger = []
        epi_losses_logger = []

        memory = deque([], maxlen=replay_capacity)

        target_net = self.make_net()
        self.copy_net(target_net, tau=1.0)
        # training loop
        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # rewards_logger = []
            losses_logger = []
            while not done:
                if render:
                    env.render()
                
                # sample step from environment and agent
                with torch.no_grad():
                    action = self.act(state, noise_scale=epsilon)
                action_scalar = action.squeeze(0).cpu().numpy()
                # print(action_scalar)
                state_new, reward, terminated, truncated, _ = env.step(action_scalar)
                reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
                # print(state_new, action_scalar, reward)
                done = terminated or truncated
                if done:
                    state_new = None
                else:
                    state_new = torch.tensor(state_new, dtype=torch.float32).unsqueeze(0).to(self.device)
                
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
                
                # update network and log
                loss = self.update(states, actions, rewards, new_states, target_net=target_net)
                losses_logger.append(loss)
                # rewards_logger.append(reward)

                # update loop invariants
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                state = state_new

            if episode % test_every == 0:
                test_reward = self.test(env, episodes=test_episodes, render=False)
                print(f'Episode {episode}: test reward {np.mean(test_reward)}')
                print(f'loss {np.mean(losses_logger, axis=0)}')
                print(f'epsilon {epsilon}')

                epi_rewards_logger.append(np.mean(test_reward))
            epi_losses_logger.append(np.mean(losses_logger))
        return epi_rewards_logger, epi_losses_logger