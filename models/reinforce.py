from .common import *

class Reinforce(BaseRLAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.policy = nn.Sequential(
            nn.Linear(self.input_shape[0], 50),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(50, self.num_actions),
            nn.Softmax(dim=1)
        )
        self.optimizer = self.optimizer(self.parameters())

    def forward(self, x):
        return self.policy(x)
    
    def act(self, state):
        # print("state: ", state, "=======")
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        probs = self.forward(state)
        action = probs.multinomial(num_samples=1).data[0].item()
        log_probs = torch.log(probs[:, action])
        return action, log_probs
    
    def update(self, rewards, log_probs):
        # calculate discounted returns from rewards
        returns = []
        G = (rewards * (self.gamma ** np.arange(len(rewards)))).sum()
        for r in rewards:
            returns.append(G)
            G -= r
            G /= self.gamma
        returns= np.array(returns)
        
        loss = - torch.from_numpy(returns) * torch.cat(log_probs)
        loss = loss.sum()
        # print(loss.shape, log_probs[i].shape, G.shape)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()