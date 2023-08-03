from .common import *

class DQN(BaseRLAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(self.input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, state, epsilon):
        if np.random.random() > epsilon:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action