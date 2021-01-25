import tensorflow_probability as tfp 
import tensorflow as tf

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs = [0.8, 0.2]) #80% chance cold, 20% chance hot
transitional_distribution = tfd.Categorical(probs = [[0.7,0.3], [0.2, 0.8]])#70% going from cold to cold, #20% from hot to cold
observation_distribution = tfd.Normal(loc = [0., 15.], scale = [5., 10.])#loc is mean and scale is standard deviation for cold and hot respective