### Reinforce
1. I can't learn per timestep in pytorch? (At least I couldn't implement it). They say that performance does not decrease much on doing this
2. Do I need to normalise reward? Does it help with variance? 

### DQN
1. Sudden drop in accuracy: https://ai.stackexchange.com/questions/28079/deep-q-learning-catastrophic-drop-reasons
    a. According to this, model overfits really quickly! And the result is pretty shitty even with the best model
2. Increasing batch size helped a lot
3. Why is batchnorm/dropout bad in rl? Why do some people say it's good? 
4. Model size! Apparently policy gradient needed much smaller models 
5. Code stuff: 

    a. deques, named tupes

    b. cloning model using load_state_dict

### A2C
1. Should I send done or terminated in TD updates?
    a. This doesn't seem to matter much: idk why(affects very less data?)
2. There's something called entropy regularization. But the algorithm is called SAC apparently
3. MC a2c converged really fast. TD-0 took much more time...

### DDPG
1. I don't understand importance sampling. I probably should spend time understanding it
2. DDPG is only used for continuous action spaces(you can't take gradient wrt a discrete action space): 
    a. Noise is continous: 
        "As detailed in the supplementary materials we used an Ornstein-Uhlenbeck process"

        Spinning up:
        
        The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred. 

    b. Q value will take actions as inputs here.
        i. Why is it good to use multiple outs for Q
3. DDPG is based on this david silver paper 
4. Lil'log: 

        One detail in the paper that is particularly useful in robotics is on how to normalize the different physical units of low dimensional features. For example, a model is designed to learn a policy with the robot’s positions and velocities as input; these physical statistics are different by nature and even statistics of the same type may vary a lot across multiple robots. Batch normalization is applied to fix it by normalizing every dimension across samples in one minibatch.
<!-- ### PPO
1.  -->
