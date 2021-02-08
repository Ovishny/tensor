#using Open AI gym to build q learning algorithm

import gym
import numpy as np 
import time

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n 
ACTIONS = env.action_space.n 

Q = np.zeros((STATES, ACTIONS)) #creates a matrix with 0 values

#constants
EPISODES = 10000 #how many times to run environment from the beginning
MAX_STEPS = 100 #max number of steps allowed for each run

LEARNING_RATE = 0.81 #learning rate
GAMMA = 0.96

RENDER = False #if want to see training set to true
#pick an action
epsilon = 0.9 #start with a 90% chance of picking a random action

rewards = []
for episode in range(EPISODES):

	state = env.reset()
	for _ in range(MAX_STEPS):

		if RENDER:
			env.render()
		#code to pick action
		if np.random.uniform(0,1) < epsilon: #check if random selected value less than epsilon
			action = env.action_space.sample()#take random action
		else:
			action = np.argmax(Q[state, :])#use q table to pick best action on current values

		next_state, reward, done, _ = env.step(action)

		#updating q values
		Q[state, action] = Q[state, action] + LEARNING_RATE*(reward + GAMMA*np.max(Q[new_state, :]) - Q[state,action])

		state = next_state

		if done:
			rewards.append(reward)
			epsilon -= 0.001
			break #reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:")
#can see q values now



