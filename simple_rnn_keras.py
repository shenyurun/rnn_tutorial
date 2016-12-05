#from __future__ import print_function
import keras
import numpy as np
import ast
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import RepeatVector, Dense, Activation
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
		y_train.append(y)

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
		y_val.append(y)

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

pdb.set_trace()

print "Building model..."
model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, input_shape=(n_steps, n_in)))
model.add(RepeatVector(n_steps))
for _ in range(LAYERS-1):
    model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(n_out)))
#model.add(Activation('softmax'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, nb_epoch=30, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
