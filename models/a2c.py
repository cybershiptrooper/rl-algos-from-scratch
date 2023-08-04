from .common import *

class A2C(BaseRLAlgorithm):
    def act(self, state, epsilon):
        raise NotImplementedError
    
    def update(self, S, A, R, S_new):
        # basic deep q learning update for a single transition:
        raise NotImplementedError
    
    def train(self, env, num_episodes=1000, max_steps=1000, render=False, verbose=False):
        raise NotImplementedError