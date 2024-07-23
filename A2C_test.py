from cartpole import CartPoleEnvironment as env
from A2C_agent import Actor, A2C_trainer

agent = Actor(input_space=4, action_space=3)
trainer = A2C_trainer(env(), actor=agent)

trainer.train(epochs=100000, ep_len=1000)
trainer.test(ep_len=1000, display=True)
