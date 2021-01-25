import tensorflow_probability as tfp 
import tensorflow as tf

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs = [0.8, 0.2]) #80% chance cold, 20% chance hot
transition_distribution = tfd.Categorical(probs = [[0.7,0.3], [0.2, 0.8]])#70% going from cold to cold, #20% from hot to cold
observation_distribution = tfd.Normal(loc = [0., 15.], scale = [5., 10.])#loc is mean and scale is standard deviation for cold and hot respective

model = tfd.HiddenMarkovModel(
	initial_distribution = initial_distribution,
	transition_distribution = transition_distribution,
	observation_distribution = observation_distribution,
	num_steps = 7 #number of days we want to observe
	)

mean = model.mean()

with tf.compat.v1.Session() as sess:
	print(mean.numpy())