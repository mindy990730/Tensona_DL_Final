from lstm_model import *
from preprocess import Data
import time
import math
# from attention import *
import numpy as np
import pickle
import linecache
import tensorflow as tf
import os
import sys
from decode import *

class encoder_params():
	def __init__(self):
		self.data_folder_path = '../data'
		self.friends_output_file_name = 'friends_transcripts.csv'

		self.lr_rate = 0.0005
		self.embed_size = 256
		self.batch_size = 64
		self.hidden_sz = 512
		self.start_halve = 5
		self.dropout = 0.1

		self.speaker_mode = True
		self.addressee_mode = True
		
		self.sentence_max_length = 20
		self.max_epochs = 1

class encode_model():
	def __init__(self, params, data, is_speaker):
		self.params = params
		self.data = data
		friends_data_dict = self.data.friends_tsv(num_seasons=10)
		self.friends_data = self.data.cleanup_and_build_dict(friends_data_dict)
		self.num_vocab = len(list(self.data.vocab_dict.keys())) # 15105
		self.num_characters = len(list(self.data.character_dict.keys())) #656
		print('num_characters = ', self.num_characters)
		print('num_vocab = ', self.num_vocab)
		self.train_data, self.test_data = self.data.train_test_split(self.friends_data, p_split=0.9) # num_train = 45416
		self.model = lstm_model(self.params, self.num_vocab, self.num_characters, is_speaker)
		# self.model = beam_decoder(self.params, self.num_vocab, self.num_characters, is_speaker)


	def train(self):
		num_epochs = 0
		initial_state = None
		while num_epochs < self.params.max_epochs:
			print('=========================EPOCH ', num_epochs, "==========================\n")
			# Adjust learning rate
			if num_epochs > self.params.start_halve:
				self.params.lr_rate *= 0.5
			# Loop through all train_data in batches
			start_index = 0
			while (start_index + self.params.batch_size) < len(self.train_data[0]):
				sources, targets, speakers, addressees = self.data.read_batch(self.train_data, start_index, mode='train')
				with tf.GradientTape() as tape:
					loss, probs = self.model.call(sources, targets, speakers, addressees, initial_state)
				print('-----------batch ', int(start_index/self.params.batch_size), ": loss = ", loss, " ---------------\n")
				gradients = tape.gradient(loss, self.model.trainable_variables)
				# print('       gradients = ', gradients)
				self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
				start_index += self.params.batch_size
			# Increment for next epoch
			num_epochs += 1
		pass

	def test(self):
		loss_list = []
		accuracy_list = []
		num_epochs = 0
		while num_epochs < self.params.max_epochs:
			start_index = 0
			while (start_index + self.params.batch_size) < len(self.test_data[0]):
			# while (start_index + self.params.batch_size) < len(self.test_data[0]):
				# Read in batched test_data
				sources, targets, speakers, addressees = self.data.read_batch(self.test_data, start_index, mode='train')
				# Forward pass to get loss
				loss, probs = self.model.call(sources, targets, speakers, addressees)
				loss_list.append(loss)
				# Calculate accuracy 
				labels = targets[:, 1:] # shape = (batch_size, sentence_max_length-1)
				labels = tf.transpose(labels) # shape = (sentence_max_length-1, batch_size)
				acc = self.model.accuracy_function(probs, labels)
				# TODO: weigh by num_valid_words: see hw4
				accuracy_list.append(acc)
				# Increment for next batch
				start_index += self.params.batch_size
			# Increment for next epoch
			num_epochs += 1
			# Show examples if reaching the end of test data
			if num_epochs == self.params.max_epochs:
				self.show_example(probs, labels)
		self.visualize_data(loss_list,mode='loss')
		self.visualize_data(accuracy_list, mode='accuracy')
		l = tf.reduce_mean(loss_list)
		perplexity = tf.math.exp(l)
		acc = tf.reduce_mean(accuracy_list)
		return perplexity, acc

	#TODO: Save the trained model so that it can be used in decoder
	def save_model(self):
		pass
	
	def show_example(self, probs, labels):
		"""
        Show examples of predicted response vs. true response
        :param probs:  a 3-D tensor that contains probabilities calculated for each column of words
                        in target, shape = (sentence_max_length-1, batch_size, num_vocab)
        :param labels: prediction of next word in target, shape = (sentence_max_length-1, batch_size)
        :return: scalar tensor of accuracy of the batch between 0 and 1     
        """
		labels = tf.transpose(labels)
		decoded_vocab_ids = tf.argmax(input=probs, axis=2) 
		decoded_vocab_ids = tf.transpose(decoded_vocab_ids) # shape = (batch_size, sentence_max_length-1)
		for row in range(self.params.batch_size - 5, self.params.batch_size, 1):
			sentence = []
			for col in range(0, tf.shape(decoded_vocab_ids)[1], 1):
				sentence.append(list(self.data.vocab_dict.keys())[list(self.data.vocab_dict.values()).index(decoded_vocab_ids[row][col])])
			print(' '.join(word for word in sentence))
			print(labels[row],'\n')
		pass

	def visualize_data(self, loss, mode='loss'):
		"""

		Takes in array of loss from each episode, visualizes loss over episodes

		:param loss: List of loss from all episodes
		"""

		x_values = arange(0, len(loss), 1)
		y_values = loss
		plot(x_values, y_values)
		xlabel('Batch')
		if mode=='loss':
			ylabel('loss')
			title('Loss by Batch')
		elif mode=='accuracy':
			ylabel('accuracy')
			title('Accuracy by Batch')
		else:
			print('graph cannot be shown')
		grid(True)
		show()


if __name__ == '__main__':
	
	if len(sys.argv) != 2 or sys.argv[1] not in {"SPEAKER", "SPEAKER_ADDRESSEE"}:
		print("USAGE: python encode.py <Model Type>")
		print("<Model Type>: [SPEAKER / SPEAKER_ADDRESSEE]")
		exit()
		
	# Initialize model
	if sys.argv[1] == "SPEAKER":
		is_speaker = True
	elif sys.argv[1] == "SPEAKER_ADDRESSEE":
		is_speaker = False


	start = time.time()
	params = encoder_params()
	data = Data(params)
	print('encode.py: created params and data')
	encode_m = encode_model(params, data, is_speaker)
	print('encode.py: created encode_model')
	encode_m.train()
	print('encode.py: finished training encode_model')
	perplexity, acc = encode_m.test()
	print('test perplexity = ', perplexity)
	print('test accuracy = ', acc)
	end = time.time()
	sec_passed = end-start
	minutes_passed = math.floor(sec_passed / 60)
	sec_left = sec_passed % 60
	print('RUNTIME = ', minutes_passed,' minutes ', sec_left, 'seconds\n')
	# model = (args)
	# model.train()