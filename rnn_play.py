#recursive neural network to generate a play
import tensorflow as tf 


import os
import numpy as np 

#load dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# #to use own data, use this code--------------------------------------
# from google.colab import files
# path_to_file = list(files.upload().keys())[0]
#---------------------------------------------------------------------------------------

#read contents of file
text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')#read and decode for py2 compat
print('Length of text: {} characters'.format(len(text)))#length of text is num of char
print(text[:250])

#need to encode the text------------------------------------
vocab = sorted(set(text))
#creating a mapping from unique charactes to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
	return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
#--------------------------------------------------------------

# print("Text: ", text[:13])
# print("Encoded: ", text_to_int(text[:13]))

#create function to go from int to text
def int_to_text(ints):
	try:
		ints = ints.numpy()
	except:
		pass
	return ''.join(idx2char[ints])

# print(int_to_text(text_as_int[:13]))

#create training examples
seq_length = 100 # length of sequence for a training example
example_per_epoch = len(text)//(seq_length + 1)

#create training examples/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
#use batch method to turn stream of characters into batches of desired length
sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)
#use sequences of length 101 and split into input and output
def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text,target_text

#use map to apply above function to every entry
dataset = sequences.map(split_input_target)

# for x,y in dataset.take(2):
# 	print("\n\nEXAMPLE\n")
# 	print("INPUT")
# 	print(int_to_text(x))
# 	print("\nOUTPUT")
# 	print(int_to_text(y))

#make training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab) #vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

BUFFER_SIZE = 10000 #buffer size to shuffle dataset. Tf data designed to work with infinite sequences. maintains a buffer in which to shuffle
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

#building a model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size,None]),
		tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True, recurrent_initializer = 'glorot_uniform'),
		tf.keras.layers.Dense(vocab_size)
	])
	return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

#create loss function
def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels,logits, from_logits=True)

#compile model
model.compile(optimizer = 'adam', loss = loss)

#creating checkpoints
#directory for saved checkpoints
checkpoint_dir = './training_checkpoints'
#name of file
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath = checkpoint_prefix,
	save_weights_only = True)

# # training the model
# history = model.fit(data, epochs = 2, callbacks = [checkpoint_callback])
# import gc
# gc.collect()

# loading the model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size = 1)

#once model finished training we can find latest checkpoints that stores model weights
print(tf.train.latest_checkpoint(checkpoint_dir))
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1,None]))

#load any checkpoint we want by specifyinf the exact file to load
# checkpoint_num = 2
# model.load_weights(tf.train.load_checkpoint("./training_checkpoints\\ckpt_" + str(checkpoint_num)))
# model.build(tf.TensorShape([1,None]))

#generate text
def generate_text(model, start_string):
	#number of characters to generate
	num_generate = 800

	#converting start string to numbers
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)

	#empty string to store results
	text_generated = []


	#low temp in more predictable text, high more surprising
	temperature = 1.0

	#batch size 1 here
	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)
		#remove batch dimension
		predictions = tf.squeeze(predictions, 0)

		#using categorical distribution to predict character returned by model
		predictions = predictions/temperature
		predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()

		#pass predicted character as next input along with previous hidden state
		input_eval = tf.expand_dims([predicted_id], 0)

		text_generated.append(idx2char[predicted_id])
	return(start_string + ''.join(text_generated))

inp = input("Type a starting_string: ")
print(generate_text(model, inp))