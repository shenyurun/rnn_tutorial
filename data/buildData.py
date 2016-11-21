import numpy as np
import nltk
import logging
import os, sys
import itertools

def load_data(filename, logger):

	vocabulary_size = 4000
	unknown_token = "UNKNOWN_TOKEN"
	sentence_start_token = "SENTENCE_START"
	sentence_end_token = "SENTENCE_END"

	logger.info("processing %s" % filename)
	tokenized_sentences, sentences = [], []
	with open(filename, 'r') as fr:
		for line in fr:
			line = line.rstrip('\n')
			if line == '':
				continue
			sentences.append(line)
			sentence = "%s %s %s" % (sentence_start_token, line, sentence_end_token)
			sentence = nltk.word_tokenize(sentence)
			tokenized_sentences.append(sentence)
	logger.info("tokenized_sentence length = %s" % len(tokenized_sentences))
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	logger.info("Found %s unique words tokens." % len(word_freq.items()))
	
	vocab = word_freq.most_common(vocabulary_size - 1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
	
	logger.info("Using vocabulary size %s." % vocabulary_size)
	logger.info("The least frequent word in our vocabulary is %s and appeared %s times" % (vocab[-1][0], vocab[-1][1]))
	
	for i, sent in enumerate(tokenized_sentences):
		tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
	
	logger.info("Example sentence: %s" % sentences[0])
	logger.info("Example sentence after Pre-processing: %s" % tokenized_sentences[0])
	
	x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
	
	logger.info("Example x_train: %s" % x_train[0])
	logger.info("Example y_train: %s" % y_train[0])
	
	return (index_to_word, vocabulary_size, x_train, y_train)
	
def save_data(index_to_word, filename):
	with open(filename, "w") as fw:
		for i, w in enumerate(index_to_word):
			fw.writelines(str(i) + "\t" + w + "\n")
	
def save_data_binary(data):
	x_train, y_train = data
	x_train.tofile("train2.bn", sep='\n')
	y_train.tofile("label2.bn", sep='\n')
	
def getLogger(name):
	FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
	logFormatter = logging.Formatter(FORMAT)
	rootLogger = logging.getLogger()
	rootLogger.setLevel(logging.DEBUG)
	
	fileHandler = logging.FileHandler("mylog2.log")
	fileHandler.setFormatter(logFormatter)
	rootLogger.addHandler(fileHandler)
	
	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	rootLogger.addHandler(consoleHandler)
	
	return rootLogger

if __name__ == '__main__':
	
	logger = getLogger("getData")
	try:
		filename = sys.argv[1]
	except:
		logger.error("not enough arguments!")
		exit(1)
		
	(index_to_word, vocabulary_size, x_train, y_train) = load_data(filename, logger)
	save_data(index_to_word, "dictionary2.txt")
	save_data_binary((x_train, y_train))
