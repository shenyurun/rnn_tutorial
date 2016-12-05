from __future__ import print_function
import keras
import numpy as np
import ast
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import RepeatVector

print "Reading training data..."

X_train = []
with open('data/train.bn', 'rb') as f:
	train_data = f.readlines()
	for x in train_data:
		x = ast.literal_eval(x)
		X_train.append(x)

y_train = []
with open('data/trainLabel.bn', 'rb') as f:
	train_label = f.readlines()
	for y in train_label:
		y = ast.literal_eval(y)
		y_train.append(y)

X_val = []
with open('data/test.bn', 'rb') as f:
	val_data = f.readlines()
	for x in val_data:
		x = ast.literal_eval(x)
		X_val.append(x)

y_val = []
with open('data/testLabel.bn', 'rb') as f:
	val_label = f.readlines()
	for y in val_label:
		y = ast.literal_eval(y)
		y_val.append(y)

n_in = 4000
n_out = 4000
n_steps = 5

HIDDEN_SIZE = 40
BATCH_SIZE = 1
LAYERS = 1

print "Building model..."
model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, input_shape=(n_steps, n_in)))
model.add(RepeatVector(n_steps))
for _ in range(LAYERS-1):
    model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(n_out)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=30, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))