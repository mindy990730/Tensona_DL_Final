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
from matplotlib import pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

class persona_params():
	def __init__(self):
		self.data_folder_path = '../data'
		self.friends_output_file_name = 'friends_transcripts.csv'

		self.lr_rate = 0.0001
		self.decay_steps = 10000
		self.decay_rate = 0.97

		self.embed_size = 256
		self.batch_size = 128
		self.hidden_sz = 512
		self.start_halve = 5
		self.dropout = 0.1

		self.speaker_mode = True
		self.addressee_mode = True
		
		self.sentence_max_length = 15
		self.max_epochs = 2
		self.EOS = 0 
		self.EOT = 1 

class persona_model():
	def __init__(self, params, data, is_speaker, is_friends):
		self.params = params
		self.data = data

		if is_friends==True:
			friends_data_dict = self.data.friends_tsv(num_seasons=3)
			self.data_dict = self.data.cleanup_and_build_dict(friends_data_dict)
			self.num_characters = self.data.num_characters
		else:
			dialogue_data_dict = self.data.dialogue_tsv()
			self.data_dict = self.data.build_dialogue_dict(dialogue_data_dict)
			self.num_characters = self.data.num_characters
		
		self.num_vocab = len(list(self.data.vocab_dict.keys())) # Friends: 15105 

		print(self.data_dict[0:5])

		print('num_characters = ', self.num_characters)
		print('num_vocab = ', self.num_vocab)
		self.train_data, self.test_data = self.data.train_test_split(self.data_dict, p_split=0.9) # Friends: num_train = 45416
		self.model = lstm_model(self.params, self.num_vocab, self.num_characters, is_speaker)
		# self.model = beam_decoder(self.params, self.num_vocab, self.num_characters, is_speaker)
		self.checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model, step = tf.Variable(1))
		if is_speaker and is_friends:
			self.manager = tf.train.CheckpointManager(self.checkpoint, '../tmp/s_F', max_to_keep = 5)
		elif is_speaker and not is_friends:			
			self.manager = tf.train.CheckpointManager(self.checkpoint, '../tmp/s_D', max_to_keep = 5)
		elif not is_speaker and is_friends:
			self.manager = tf.train.CheckpointManager(self.checkpoint, '../tmp/s_a_F', max_to_keep = 5)
		else:
			self.manager = tf.train.CheckpointManager(self.checkpoint, '../tmp/s_a_D', max_to_keep = 5)


	def train(self, is_test):
		num_epochs = 0
		initial_state = None
		while num_epochs < self.params.max_epochs:
			print('=========================EPOCH ', num_epochs, "==========================\n")
			# Adjust learning rate
			# if num_epochs > self.params.start_halve:
			# 	self.params.lr_rate *= 0.5
			# Loop through all train_data in batches
			start_index = 0
			# if os.path.isdir('../saved_weights') and is_test:
			# 	self.model.encoder.load_weights('../saved_weights/en_weights.tf')
			# 	self.model.decoder.load_weights('../saved_weights/de_weights.tf')
			# 	print('Loaded existing weights.')
			while (start_index + self.params.batch_size) < len(self.train_data[0]):
				if self.manager.latest_checkpoint:
					print("Restored from {}".format(self.manager.latest_checkpoint))
				else:
					print("Initializing from scratch.")
				sources, targets, speakers, addressees = self.data.read_batch(self.train_data, start_index, mode='train')
				with tf.GradientTape() as tape:
					loss, probs = self.model.call(sources, targets, speakers, addressees, initial_state)
				print('-----------------------------batch ', int(start_index/self.params.batch_size), ": loss = ", loss, " ---------------------\n")
				gradients = tape.gradient(loss, self.model.trainable_variables)
				# print('       gradients = ', gradients)
				self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
				self.checkpoint.step.assign_add(1)
				if int(self.checkpoint.step) % 10 == 0:
					save_path = self.manager.save()
					print("Saved checkpoint for step ", format(int(self.checkpoint.step)), ": ", save_path, )
				if start_index==0:
					print(self.model.encoder.summary())
					print(self.model.decoder.summary())
				start_index += self.params.batch_size
				if is_test and start_index == self.params.batch_size*3:
					return initial_state
			# Increment for next epoch
			num_epochs += 1
		return initial_state

	def test(self, initial_state):
		loss_list = []
		accuracy_list = []
		num_epochs = 0
		valid_words = []
		while num_epochs < self.params.max_epochs:
			start_index = 0
			while (start_index + self.params.batch_size) < len(self.test_data[0]):
			# while (start_index + self.params.batch_size) < len(self.test_data[0]):
				# Read in batched test_data
				sources, targets, speakers, addressees = self.data.read_batch(self.test_data, start_index, mode='train')
				# Forward pass to get loss
				loss, probs = self.model.call(sources, targets, speakers, addressees, initial_state)
				loss_list.append(loss)
				#Count valid words
				mask = np.ones((self.params.batch_size, self.params.sentence_max_length), dtype=int)
				for row in range(self.params.batch_size):
					for col in range(self.params.sentence_max_length):
						if targets[row][col]==self.params.EOS or targets[row][col]==self.params.EOT:
							mask[row][col]=0
				mask = tf.convert_to_tensor(mask, dtype=tf.float32)
				# Compute number of words in this batch
				num_valid_words = tf.reduce_sum(mask)
				valid_words.append(num_valid_words)
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
		# Compute per-word perplexity
		total_words = tf.reduce_sum(valid_words)
		l = tf.reduce_sum(loss_list)
		l = tf.divide(l, total_words)
		perplexity = tf.math.exp(l)
		acc = tf.reduce_sum(accuracy_list)
		acc = tf.divide(acc, total_words)
		return perplexity, acc


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
		num = 0	
		all_scores = []
		for row in range(0, self.params.batch_size, 1):
			sentence = []
			ans = []
			for col in range(0, tf.shape(decoded_vocab_ids)[1], 1):
				sentence.append(list(self.data.vocab_dict.keys())[list(self.data.vocab_dict.values()).index(decoded_vocab_ids[row][col])])
				ans.append(list(self.data.vocab_dict.keys())[list(self.data.vocab_dict.values()).index(labels[row][col])])
			if num >= self.params.batch_size -5:
				print('\nSample %d: ------------------------------------------------------------------------------'%num)
				print(' '.join(word for word in sentence))
				print(' '.join(word for word in ans))
			for i in range(len(sentence)):
				all_scores.append(sentence_bleu(sentence[i], ans[i]))
			num += 1
		print('ave bleu score = ', tf.reduce_mean(all_scores))
		pass

	def visualize_data(self, loss, mode='loss'):
		"""

		Takes in array of loss from each episode, visualizes loss over episodes

		:param loss: List of loss from all episodes
		"""

		x_values = np.arange(0, len(loss), 1)
		y_values = loss
		plt.plot(x_values, y_values)
		plt.xlabel('Batch')
		if mode=='loss':
			plt.ylabel('loss')
			plt.title('Loss by Batch')
		elif mode=='accuracy':
			plt.ylabel('accuracy')
			plt.title('Accuracy by Batch')
		else:
			print('graph cannot be shown')
		plt.grid(True)
		plt.show()

	def sentence_bleu_score(self, reference, prediction): 
		"""

		Caclculates the BLEU score for a given sentence

		:param reference: a list of reference sentences where each reference is a list of tokens
		:param prediction: a list of predicted sentences where each prediction is a list of tokens
		"""
		# e.g.
		# reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
		# predicted_response = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
		score = sentence_bleu(reference, prediction)
		return score

	def corpus_bleu_score(self, reference, prediction): 
		"""

		Caclculates the BLEU score for a given corpus

		:param reference: a list of documents where each document is a list of references and each alternative reference is a list of tokens
		:param prediction: a list where each document is a list of tokens, e.g. a list of lists of tokens
		"""
		# e.g.
		# reference = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
		# predicted_response = [['this', 'is', 'a', 'test']]
		score = corpus_bleu(reference, prediction)
		return score

if __name__ == '__main__':
	
	if len(sys.argv) != 4 or sys.argv[1] not in {"SPEAKER", "SPEAKER_ADDRESSEE"} or sys.argv[2] not in {"FRIENDS", "DIALOGUE"} or sys.argv[3] not in {"TEST", "TRAIN"}:
		print("USAGE: python personal_model.py <Model Type> <Dataset>")
		print("<Model Type>: [SPEAKER / SPEAKER_ADDRESSEE]")
		print("<Model Type>: [FRIENDS / DIALOGUE]")
		exit()
		
	# Initialize model
	if sys.argv[1] == "SPEAKER":
		is_speaker = True
	elif sys.argv[1] == "SPEAKER_ADDRESSEE":
		is_speaker = False
	
	if sys.argv[2] == "FRIENDS":
		is_friends = True
	elif sys.argv[2] == "DIALOGUE":
		is_friends = False

	if sys.argv[3] =="TEST":
		is_test = True
	else:
		is_test = False


	start = time.time()
	params = persona_params()
	data = Data(params)
	print('personal_model.py: created params and data')
	persona_m = persona_model(params, data, is_speaker, is_friends)
	print('personal_model.py: created persona_model')
	initial_state = persona_m.train(is_test)
	print('personal_model.py: finished training persona_model')
	perplexity, acc = persona_m.test(initial_state)
	print('test perplexity = ', perplexity)
	print('test accuracy = ', acc)
	end = time.time()
	sec_passed = end-start
	minutes_passed = math.floor(sec_passed / 60)
	sec_left = sec_passed % 60
	print('RUNTIME = ', minutes_passed,' minutes ', sec_left, 'seconds\n')
	persona_m.model.encoder.save_weights('../saved_weights/en_weights.tf', save_format='tf')
	persona_m.model.decoder.save_weights('../saved_weights/de_weights.tf', save_format='tf')
	
	# model = (args)
	# model.train()