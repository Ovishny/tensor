#Using DNN(Deep Neural Networks) classifier to predict the species of flower based on its characteristics

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
#defining the input function
def input_fn(features, labels, training = True, batch_size = 256):
	#convert input to dataset
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	#shuffle and repeat if training
	if training:
		dataset = dataset.shuffle(1000).repeat()

	return dataset.batch(batch_size)

#create feature columns
my_feature_columns = []
for key in train.keys():
	my_feature_columns.append(tf.feature_column.numeric_column(key = key))

#build DNN with 2 hidden layers with 30 and 10 hidden nodes each

classifier = tf.estimator.DNNClassifier(feature_columns = my_feature_columns,
	#two hidden layers of 30 and 10 nodes
	hidden_units = [30,10],
	#model must choose between three classes
	n_classes = 3)

#train the model we just created
classifier.train(
	input_fn = lambda: input_fn(train, train_y, training = True),
	steps = 5000)

#evaluates accuracy of model
eval_result = classifier.evaluate(input_fn = lambda: input_fn(test, test_y, training = False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#this will allow a user to input info --------------------------------------------
def input_fn(features, batch_size = 256):
	#convert input to dataset without labels
	return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
	valid = True
	while valid:
		val = input(feature + ": ")
		if not val.isdigit(): valid = False
	predict[feature] = [float(val)]

predictions = classifier.predict(input_fn = lambda: input_fn(predict))
for pred_dict in predictions:
	class_id = pred_dict['class_ids'][0]
	probability = pred_dict['probabilities'][class_id]

	print('Prediction is "{}" ({:.1f}%)'.format(
		SPECIES[class_id], 100*probability))
