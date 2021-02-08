import gym
import numpy as np 
import time

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n 
ACTIONS = env.action_space.n 

Q = np.zeros((STATES, ACTIONS))