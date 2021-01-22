#Using DNN(Deep Neural Networks) to predict the species of flower based on its characteristics

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf 
import pandas as pd 

#define column names and species types
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
#grab csv files from link and adds it to local desktop, and makes a path to it
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
#takes the csv files and turns them into panda dataframes
train = pd.read_csv(train_path, names = CSV_COLUMN_NAMES, header = 0)
test = pd.read_csv(test_path, names = CSV_COLUMN_NAMES, header = 0)
#pop out the column we are trying to predict
train_y = train.pop('Species')
test_y = test.pop('Species')
