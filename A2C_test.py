from environments.cartpole import CartPoleEnvironment as env
#import environments.pendulum as env
#from environments.flappy_bird import FlappyBirdEnvironment as env

#import environments.cliffwalk as env
from A2C_agent import Actor, A2C_trainer

import torch

#game.setR(0)

def plot(a):
    lst = []
    for line in a.split('\n'):
        if "test score:" in line: lst.append(float(line.split(' ')[-1]))

    temp = lst
    lst = []
    import numpy as np
    for i in range(0,len(temp),20):
        lst.append(np.average(temp[i:i+20]))

    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(lst))],lst)
    plt.show()

agent = Actor(input_space=4, action_space=3)
trainer = A2C_trainer(env(), actor=agent)

trainer.train(epochs=100000, ep_len=1000)
trainer.test(ep_len=1000, display=True)
