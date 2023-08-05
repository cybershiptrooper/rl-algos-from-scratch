from .common import *

class Reinforce(BaseRLAlgorithm):
    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)
    
    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        probs = self.forward(state)
        action = probs.multinomial(num_samples=1).data[0].item()
        log_probs = torch.log(probs[:, action])
        return action, log_probs
    
    def update(self, rewards, log_probs):
        # calculate discounted returns from rewards
        for i in range(-2, -len(rewards), -1):
            rewards[i] += self.gamma * rewards[i+1] # discounted return: rewards[i] = G_t
        # normalize rewards: 
        #  Ref 1: https://stackoverflow.com/questions/49801638/normalizing-rewards-to-generate-returns-in-reinforcement-learning
        #  Ref 2: https://ai.stackexchange.com/questions/16136/should-the-policy-parameters-be-updated-at-each-time-step-or-at-the-end-of-the-e
        returns = (rewards - rewards.mean())/ (rewards.std() + 1e-9) 
        
        if self.device.type == "mps":
            # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        else:
            returns = torch.tensor(returns).to(self.device)
        # calculate policy gradient
        loss = - returns * torch.cat(log_probs)
        loss = loss.sum()
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, env, episodes=800, render=False):
        # set up
        epi_rewards_logger = []
        epi_losses_logger = []

        for episode in tqdm(range(1, episodes+1)):
            done = False
            state, _ = env.reset()
            rewards = []
            log_probs = []

            # sample an episode
            while not done:
                if render:
                    env.render()
                state = state.astype(np.float32)
                action, log_prob = self.act(state)
                log_probs.append(log_prob)
                # state, reward, terminated, truncated, info 
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards.append(reward)
            env.reset()

            # update the model
            rewards_arr = np.array(rewards)
            loss = self.update(rewards_arr, log_probs)

            # log
            epi_losses_logger.append(loss)
            epi_rewards_logger.append(np.sum(rewards))
            if(episode % 100 == 0 and episode >= 100):
                print("Episode: ", episode, 
                      "\nMean Reward for last 100 episodes: ", np.mean(epi_rewards_logger[-100:]))
        return np.array(epi_rewards_logger), np.array(epi_losses_logger)