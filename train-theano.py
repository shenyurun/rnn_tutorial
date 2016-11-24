#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from rnn_theano import gradient_check_theano
import pdb
import ast

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '10'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '4'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.5'))
_NEPOCH = int(os.environ.get('NEPOCH', '3'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

def train_with_sgd(model, X_all, y_all, learning_rate=0.005, nepoch=1, evaluate_loss_after=1):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0

#    pdb.set_trace()
    num_samples = len(X_all)
    num_train = int(num_samples * 0.8)
    X_train = X_all[0:num_train]
    y_train = y_all[0:num_train]
    X_val = X_all[num_train:num_samples]
    y_val = y_all[num_train:num_samples]

    for epoch in range(nepoch):
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.agd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
        
        
#        gradient_check_input(model, X_train[100], y_train[100], h=0.001, error_threshold=0.01)
        
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            # Output part of training samples
            for x,y in zip(X_train[0:10],y_train[0:10]):
                pred = model.predict(x)
                print("train label: %s" % y)
                print("train prediction: %s" % pred)
    
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch+1, loss)
          
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            
            # ADDED! Saving model oarameters
#           save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        
        # Validation
        loss = model.calculate_loss(X_val, y_val)
        print "Loss on validation: %f" % loss
        perplex = model.calculate_perplexity(X_val, y_val)
        print "Perplexity on validation: %f" % perplex
        accuracy = model.calculate_accuracy(X_val, y_val)
        print "Accuracy on validation: %f" % accuracy

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
# print "Reading CSV file..."
# with open('data/reddit-comments-2015-08.csv', 'rb') as f:
#     reader = csv.reader(f, skipinitialspace=True)
#     reader.next()
#     # Split full comments into sentences
#     sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
#     # Append SENTENCE_START and SENTENCE_END
#     sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
# print "Parsed %d sentences." % (len(sentences))
    
# # Tokenize the sentences into words
# tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# # Count the word frequencies
# word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
# print "Found %d unique words tokens." % len(word_freq.items())

# # Get the most common words and build index_to_word and word_to_index vectors
# vocab = word_freq.most_common(vocabulary_size-1)
# index_to_word = [x[0] for x in vocab]
# index_to_word.append(unknown_token)
# word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# print "Using vocabulary size %d." % vocabulary_size
# print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# # Replace all words not in our vocabulary with the unknown token
# for i, sent in enumerate(tokenized_sentences):
#     tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# # Create the training data
# X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
# y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#print "Reading training data..."
#X_train = []
#with open('data/train2.bn', 'rb') as f:
#   train_data = f.readlines()
#   for x in train_data:
#       x = ast.literal_eval(x)
#       X_train.append(x)
#y_train = []
#with open('data/label2.bn', 'rb') as f:
#   train_label = f.readlines()
#   for y in train_label:
#       y = ast.literal_eval(y)
#       y_train.append(y)


# Temporarily use a small part of training samples
# X_train = X_train[::100]
# y_train = y_train[::100]

x = np.random.randint(10, size=10, dtype='int32')
y = np.random.randint(10, size=10, dtype='int32')

#pdb.set_trace()

# Train rnn model
model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM, bptt_truncate=10)
gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01)
#t1 = time.time()
#model.s#agd_step(X_train[10], y_train[10], _LEARNING_RATE)
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

#if _MODEL_FILE != None:
#   load_model_parameters_theano(_MODEL_FILE, model)

#train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)


#gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01)

# Test the model
# print "Reading test data..."
# X_test = []
# with open('data/test.bn', 'rb') as f:
#     test_data = f.readlines()
#     for x in test_data:
#         x = ast.literal_eval(x)
#         X_test.append(x)
# y_test = []
# with open('data/testLabel.bn', 'rb') as f:
#     test_label = f.readlines()
#     for y in test_label:
#         y = ast.literal_eval(y)
#         y_test.append(y)

# To be removed
# X_test = X_test[::100]
# y_test = y_test[::100]

# Evaluate the model
# Cross entropy
# loss = model.calculate_loss(X_test, y_test)
# print "Loss: %f" % loss

# Bigram perplexity
# perplex = model.calculate_perplexity(X_test, y_test)
# print "Perplexity: %f" % perplex

# Accurary
# accuracy = model.calculate_accuracy(X_test, y_test)
# print "Accuracy: %f" % accuracy

# for x,y in zip(X_test,y_test):
#     pred = model.predict(x)
#     print "test label: "
#     print(y[:])
#     print "test prediction: "
#     print(pred[:])
