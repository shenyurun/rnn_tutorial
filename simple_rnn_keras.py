#from __future__ import print_function
import keras
import numpy as np
import ast
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, RepeatVector, TimeDistributedDense, Activation
from keras.layers.wrappers import TimeDistributed
import pdb

print "Reading training data..."

X_train = []
with open('data/train.bn', 'rb') as f:
	train_data = f.readlines()
	for x in train_data:
		x = ast.literal_eval(x)
		onehot_x = np.zeros((len(x),4000))
   		onehot_x[np.arange(len(x)), x] = 1
		X_train.append(onehot_x)

y_train = []
with open('data/trainLabel.bn', 'rb') as f:
	train_label = f.readlines()
	for y in train_label:
		y = ast.literal_eval(y)
    		onehot_y = np.zeros((len(y),4000))
    		onehot_y[np.arange(len(y)), y] = 1
		y_train.append(onehot_y)

X_val = []
with open('data/test.bn', 'rb') as f:
	val_data = f.readlines()
	for x in val_data:
		x = ast.literal_eval(x)
    		onehot_x = np.zeros((len(x),4000))
   		onehot_x[np.arange(len(x)), x] = 1
		X_val.append(onehot_x)

y_val = []
with open('data/testLabel.bn', 'rb') as f:
	val_label = f.readlines()
	for y in val_label:
		y = ast.literal_eval(y)
    		onehot_y = np.zeros((len(y),4000))
    		onehot_y[np.arange(len(y)), y] = 1
		y_val.append(onehot_y)

X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

n_in = 4000
n_out = 4000
n_steps = 5

HIDDEN_SIZE = 40
BATCH_SIZE = 1
LAYERS = 1

print "Building model..."
model = Sequential()

model.add(SimpleRNN(HIDDEN_SIZE, input_dim=n_in))

#model.add(Dense(HIDDEN_SIZE, activation="relu"))

model.add(RepeatVector(n_steps))
    
for _ in range(LAYERS-1):
    model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=True))

model.add(TimeDistributed(Dense(n_out)))

model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="sgd")

print(model.summary())

pdb.set_trace()

model.fit(X_train, y_train, nb_epoch=30, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
