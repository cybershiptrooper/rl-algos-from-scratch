from .common import *
from collections import namedtuple

ac_pair = namedtuple('ac_pair', ('actor', 'critic'))

class A2C(BaseRLAlgorithm):

    def make_net(self):
        if self.transpose:
            raise NotImplementedError
        else:
            actor = nn.Sequential(
                    nn.Linear(self.input_shape[0], self.config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                    nn.ReLU(),
                    # nn.Dropout(p=0.6),
                    nn.Linear(self.config.hidden_size, self.num_actions),
                    nn.Softmax(dim=-1)
            )

            critic = nn.Sequential(
                nn.Linear(self.input_shape[0], self.config.hidden_size),
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
    
    def act(self, state):
        probs = self.net.actor(state)
        action = probs.multinomial(num_samples=1).data[0].item()
        log_probs = torch.log(probs[:, action])
        return action, log_probs
    
    def td0_update(self, S, R, S_new, log_probs, done):
        # calculate loss
        with torch.no_grad():
            if done:
                target = torch.tensor(R, dtype=torch.float32).to(self.device).view(-1, 1)
            else:
                target = R + self.gamma * self.net.critic(S_new)
        baseline = self.net.critic(S)
        delta = target - baseline
        
        actor_loss = - log_probs * delta.detach()
        critic_loss = delta**2

        # update
        self.optimizer.actor.zero_grad()
        self.optimizer.critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_value_(self.net.actor.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.net.critic.parameters(), 100)

        self.optimizer.actor.step()
        self.optimizer.critic.step()
        return [actor_loss.mean().item(), critic_loss.mean().item()]
    
    def mc_update(self, states, rewards, log_probs):
        # calculate discounted returns from rewards
        for i in range(-2, -len(rewards), -1):
            rewards[i] += self.gamma * rewards[i+1] # discounted return: rewards[i] = G_t
        # normalize rewards: 
        #  Ref 1: https://stackoverflow.com/questions/49801638/normalizing-rewards-to-generate-returns-in-reinforcement-learning
        #  Ref 2: https://ai.stackexchange.com/questions/16136/should-the-policy-parameters-be-updated-at-each-time-step-or-at-the-end-of-the-e
        returns = (rewards - rewards.mean())/ (rewards.std() + 1e-9) 
        delta = returns - self.net.critic(states)
        actor_loss = - log_probs * delta.detach()
        critic_loss = delta**2

        # update
        self.optimizer.actor.zero_grad()
        self.optimizer.critic.zero_grad()
        actor_loss.mean().backward()
        critic_loss.mean().backward()

        # clip gradients
        torch.nn.utils.clip_grad_value_(self.net.actor.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.net.critic.parameters(), 100)

        self.optimizer.actor.step()
        self.optimizer.critic.step()
        return [actor_loss.mean().item(), critic_loss.mean().item()]

    def update(self, *args, **kwargs):
        if self.config.mc:
            return self.mc_update(*args, **kwargs)
        else:
            return self.td0_update(*args, **kwargs)
    
    def train(self, env, episodes=500, render=False):
        # set up
        epi_rewards_logger = []
        epi_losses_logger = []
        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device).view(1, -1)
            reward_logger = 0
            loss_logger = []
            if self.config.mc:
                rewards = []
                states = []
                log_probs = []
            while not done:
                if render:
                    env.render()
                action, log_prob = self.act(state)
                state_new, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                reward_logger += reward
                state_new = torch.tensor(state_new, dtype=torch.float32).to(self.device).unsqueeze(0)
                if not self.config.mc:
                    loss = self.update(state, reward, state_new, log_probs, done)
                    loss_logger.append(loss)
                else:
                    rewards.append(reward)
                    states.append(state)
                    log_probs.append(log_prob)
                state = state_new
            if self.config.mc:
                states = torch.cat(states)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                log_probs = torch.cat(log_probs)
                loss = self.update(state, rewards, log_probs)
                epi_losses_logger.append(loss)
            else: 
                epi_losses_logger.append(np.mean(loss_logger, axis = 0))
            epi_rewards_logger.append(reward_logger)
            if episode % 100 == 0 and episode > 0:
                print(f"Episode {episode}",
                      "\nMean Reward for last 100 episodess {:.2f}".format(
                            np.mean(epi_rewards_logger[-100:])
                    ))
        return epi_rewards_logger, epi_losses_logger