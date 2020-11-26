from lstm_model import *
import argparse
from preprocess import Data
import tensorflow as tf
import numpy as np

class encoder_params():
	def __init__(self):
		self.data_folder_path = '../data'
		self.friends_output_file_name = 'friends_transcripts.csv'

		self.lr_rate = 0.01
		self.embed_size = 256
		self.batch_size = 256
		self.hidden_sz = 512
		self.start_halve = 5
		self.dropout = 0.2

		self.speaker_mode = True
		self.addressee_mode = True
		
		self.sentence_max_length = 50
		self.max_epochs = 10
		
def build_args():
	parser = argparse.ArgumentParser()
	if True:
		parser.add_argument('--data_folder_path', type=str, default='data/testing',
							help='the folder that contains your dataset and vocabulary file')
		parser.add_argument('--data_file_name', type=str, default='data.txt')
		parser.add_argument('--train_file', type=str, default='train.txt')
		parser.add_argument('--dev_file', type=str, default='valid.txt')
		parser.add_argument('--dictPath', type=str, default='vocabulary')
		parser.add_argument('--save_folder', type=str, default='save/testing')
		parser.add_argument('--save_prefix', type=str, default='model')
		parser.add_argument('--save_params', type=str, default='params')
		parser.add_argument('--output_file', type=str, default='log')
		parser.add_argument('--no_save', action='store_true')
		parser.add_argument('--cpu', action='store_true')

		parser.add_argument('--UNK',type=int,default=0,
							help='the index of UNK. UNK+special_word=3.')
		parser.add_argument('--special_word', type=int, default=3,
							help='default special words include: padding, EOS, EOT.')

		parser.add_argument('--fine_tuning', action='store_true')
		parser.add_argument('--fine_tunine_model', type=str, default='model')

	parser.add_argument('--lr_rate', type=float, default=0.01)
	parser.add_argument('--embed_size', type=int, default=256)
	parser.add_argument('--SpeakerMode', action='store_true')
	parser.add_argument('--AddresseeMode', action='store_true')

	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--sentence_max_length", type=int, default=50)
	# parser.add_argument("--target_max_length", type=int, default=50)
	parser.add_argument("--max_iter", type=int, default=10)

	parser.add_argument("--hidden_sz", type=int, default=512)
	parser.add_argument("--num_layers", type=int, default=4)
	parser.add_argument("--init_weight", type=float, default=0.1)

	# parser.add_argument("--alpha", type=int, default=1)
	parser.add_argument("--start_halve", type=int, default=6)
	parser.add_argument("--thres", type=int, default=5)
	parser.add_argument("--dropout", type=float, default=0.2)


	args = parser.parse_args()
	print(args)
	print()


class encode_model():
	def __init__(self, params, data):
		self.params = params
		self.data = data
		friends_data_dict = self.data.friends_tsv(num_seasons=10)
		self.friends_data = self.data.cleanup_and_build_dict(friends_data_dict)
		self.num_vocab = len(list(self.data.vocab_dict.keys()))
		self.num_characters = len(list(self.data.character_dict.keys()))
		self.train_data, self.test_data = self.data.train_test_split(self.friends_data, p_split=0.9)
		self.model = MyLstm(self.params, self.num_vocab)

	def train(self):
		num_epochs = 0
		while num_epochs < self.params.max_epochs:
			# Adjust learning rate
			if num_epochs > self.params.start_halve:
				self.params.lr_rate *= 0.5
			# Loop through all train_data in batches
			start_index = 0
			while (start_index + self.params.batch_size) < len(self.train_data):
				sources, targets, speakers, addressees = self.data.read_batch(self.train_data, start_index, mode='train')
				with tf.GradientTape() as tape:
					loss, probs = self.model.call(sources, targets, speakers, addressees)
				gradients = tape.gradient(loss, self.model.trainable_variables)
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
			while (start_index + self.params.batch_size) < len(self.test_data):
				# Read in batched test_data
				sources, targets, speakers, addressees = self.data.read_batch(self.test_data, start_index, mode='train')
				# Forward pass to get loss
				loss, probs = self.model.call(sources, targets, speakers, addressees)
				loss_list.append(loss)
				# Calculate accuracy 
				labels = targets[:, 1:] # shape = (batch_size, sentence_max_length-1)
				labels = tf.tranpose(labels) # shape = (sentence_max_length-1, batch_size)
				acc = self.model.accuracy_function(probs, labels)
				# TODO: weigh by num_valid_words: see hw4
				accuracy_list.append(acc)
				# Increment for next batch
				start_index += self.params.batch_size
			# Increment for next epoch
			num_epochs += 1
		l = tf.reduce_mean(loss_list)
		perplexity = tf.math.exp(l)
		acc = tf.reduce_mean(accuracy_list)
		return perplexity, acc

	#TODO: Save the trained model so that it can be used in decoder
	def save_model(self):
		pass
	
if __name__ == '__main__':
	params = encoder_params()
	data = Data(params)
	print('encode.py: created params and data')
	encode_m = encode_model(params, data)
	print('encode.py: created encode_model')
	encode_m.train()
	print('encode.py: finished training encode_model')
	perplexity, acc = encode_m.test()
	print('test perplexity = ', perplexity)
	print('test accuracy = ', acc)
	# model = (args)
	# model.train()