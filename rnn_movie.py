#Natural Language Processing on Movie Revies using recurrent neural network

from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf 
import os 
import numpy as np 

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)