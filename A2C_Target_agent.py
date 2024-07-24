import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch
import torch.optim as optim

import numpy as np, scipy
import random, time, math

import matplotlib.pyplot as plt


class Critic():

    def __init__(self, input_space, discount):
        self.network = nn.Sequential(
            nn.Linear(input_space, 36),
            nn.Mish(),
            nn.Linear(36,36),
            nn.Mish(),
            nn.Linear(36,1)
            )
        self.target_network = nn.Sequential(
            nn.Linear(input_space, 36),
            nn.Mish(),
            nn.Linear(36,36),
            nn.Mish(),
            nn.Linear(36,1)
            )

        self.n_step = 5

        self.target_network.load_state_dict(self.network.state_dict())
        self.tau = .01

        self.loss_fn = torch.nn.MSELoss()
        self.opt = optim.Adam(self.network.parameters(), lr=.001, weight_decay=1e-5)
        self.discount = discount

    def soft_update(self):
        for target_param, online_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, observations, use_target=False):
        observations = torch.squeeze(observations)
        return self.network(observations)

    def update(self, observations, rewards, dones):
        self.opt.zero_grad()
        outputs = self.forward(observations)
        # GAE: A(s,a) = r + yV(s') - V(s)
        with torch.no_grad():
            targets = self.forward(observations,use_target=True).detach().clone()
        run = rewards
        for i in range(self.n_step):
            targets = torch.cat((targets[1:], torch.tensor([[0.0]], dtype=torch.float32))) * self.discount
            targets *= dones
        targets += run
        for i in range(self.n_step-1):
            run = torch.cat((run[1:], torch.tensor([[0.0]], dtype=torch.float32))) * self.discount
            run *= dones
            targets += run
        #targets = self.forward(observations,use_target=True).detach().clone()
        #targets = torch.cat((targets[1:], torch.tensor([[0.0]], dtype=torch.float32))) * self.discount
        #targets *= dones # apply dones mask so episode doesnt affect earlier one
        #targets += rewards
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.opt.step()

        self.soft_update()

        td_e = targets.detach()-outputs.detach()
        return td_e
        

class Actor():

    def __init__(self, input_space, action_space, network=None):

        if network == None:
            self.network = nn.Sequential(
                          nn.Linear(input_space,36),
                          nn.Mish(),
                          nn.Linear(36,36),
                          nn.Mish(),
                          nn.Linear(36,action_space),
                          nn.Softmax()
                        )
        else: self.network = network

        self.entropy_coeff = .001

        self.input_space = input_space
        self.action_space = [i for i in range(action_space)]

    def get_action(self, state, valid_actions):
        with torch.no_grad():
            action_probabilities = self.network(self.conv(state)).detach().tolist()[0]
        action_probabilities = np.multiply(action_probabilities, valid_actions)
        action = np.random.choice(self.action_space, p=np.divide(action_probabilities,sum(action_probabilities)))
        return action, action_probabilities

    def learn(self, state, action, prob, r, scale):
        #output = self.network(self.conv(state))
        output = self.network(state)
        grad = [0 for i in range(len(self.action_space))]
        entropy = sum(scipy.special.entr(output.detach().squeeze().numpy()))
        grad[action] = (-r-entropy*self.entropy_coeff) / max(prob,0.05) * scale
        grad = torch.tensor([grad], dtype=torch.float32)
        output.backward(grad)

    def set_conv(self, conv):
        self.conv = conv

    def get_optimizer(self):
        return self.network



'''
required functions in env:
    - resetEnv(), returns [state, valid_actions]
    - nextFrame(action), returns [next_state, r, valid_actions ,done]
    - convState(state), return [converted_state]

'''

class A2C_trainer():

    def __init__(self, env, actor, lr = .0003,
                 batch_size = 128, discount = .95):
        self.batch_size = batch_size
        self.discount = discount
        self.actor = actor
        self.critic = Critic(actor.input_space, discount)
        self.env = env
        self.actor.set_conv(self.env.convState)
                 
        self.opt = optim.RMSprop(self.actor.network.parameters(), lr=lr, weight_decay=1e-5)

    def train(self, epochs, ep_len, verbose=True):
        for batch in range(epochs//self.batch_size):
            self.opt.zero_grad()
            # [ state, action, action_probability, r, done ]
            hist = [[],[],[],[],[]]
            # play episode
            for ep in range(self.batch_size):
                [x.extend(y) for x,y in zip(hist,self.run_episode(ep_len))]
                if ep/self.batch_size//.1 > (ep-1)/self.batch_size//.1: print('#',end='')

            # critic update
            state_stack = torch.stack(hist[0], dim=0)
            rewards_stack = torch.tensor(hist[3]).reshape(-1,1)
            dones_stack = torch.tensor([hist[4]]).reshape(-1,1)
            advantages = self.critic.update(state_stack, rewards_stack, dones_stack).squeeze()

            # normalize advantages
            lst = advantages.numpy()
            lst = (lst-np.mean(lst)) / (np.std(lst) + 1e-10)
            hist[3] = list(lst)

            # policy gradient
            for info in list(zip(*hist)):
                state, action, action_probability, r, done = info
                self.actor.learn(state, action, action_probability, r, scale=1/len(hist[0]))
            self.opt.step()
                 
            if verbose: print(f" test score: {round(self.test(ep_len,display=False),3)}")

    def run_episode(self, ep_len):
        # [ state, action, action_probability, r, done ]
        hist = [[],[],[],[],[]]
        state, valid_actions = self.env.resetEnv()
        done = False
        for step in range(ep_len):
            action, action_probabilities = self.actor.get_action(state, valid_actions)
            next_state, r, valid_actions, done = self.env.nextFrame(action)
            [x.append(y) for x,y in zip(hist,[self.actor.conv(state), action, action_probabilities[action], r, int(not done)])]
            if done: break
            state = next_state

        # discount rewards
        r = hist[3]
        hist[3] = scipy.signal.lfilter([1], [1, -self.discount], x=r[::-1])[::-1]

        return hist

    def test(self, ep_len, display):
        done = False
        total_r = 0
        state, valid_actions = self.env.resetEnv()
        for step in range(ep_len):
            action, action_probabilities = self.actor.get_action(state, valid_actions)
            state, r, valid_actions, done = self.env.nextFrame(action,display=display)
            if display:
                with torch.no_grad():
                    print(round(self.critic.forward(self.actor.conv(state)).item(),2))
            total_r += r
            if done: break
        return total_r
        
        
        
            

        
        
        

    
        
        
        

