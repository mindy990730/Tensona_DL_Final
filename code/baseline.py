import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers 
from preprocess import Data

class baseline_params():
	def __init__(self):
		self.data_folder_path = '../data'
		self.friends_output_file_name = 'friends_transcripts.csv'

		self.lr_rate = 0.0005
		self.embed_size = 256
		self.batch_size = 128
		self.hidden_sz = 512
		self.start_halve = 5
		self.dropout = 0.1

		self.speaker_mode = True
		self.addressee_mode = True
		
		self.sentence_max_length = 20
		self.max_epochs = 1

class baseline_model(tf.keras.Model):
    def __init__(self, num_vocab, params):
        super(baseline_model, self).__init__()

        self.en_embedding = layers.Embedding(num_vocab, params.embed_size)
        self.encoder = layers.LSTM(params.hidden_sz, return_state=True)
        self.de_embedding = layers.Embedding(num_vocab, params.embed_size)
        self.decoder = layers.LSTM(params.hidden_sz, return_sequences=True, return_state=True)
        self.dense = layers.Dense(num_vocab, activation='softmax')
        self.optimizer = tf.keras.optimizers.Adam(params.lr_rate)
    
    def call(self, encoder_input, decoder_input):
        encoder_input_embedded = self.en_embedding(encoder_input)
        encoder_outputs , state_h , state_c = self.encoder(encoder_input_embedded, initial_state = None)
        state = [state_h , state_c]

        decoder_input_embedded = self.de_embedding(decoder_input)
        decoder_outputs, de_state_h, de_state_c = self.decoder(decoder_input_embedded, initial_state = state)
        output = self.dense(decoder_outputs)

        return output

    def loss_function(self, prbs, labels):

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)

        return tf.reduce_sum(loss)
    
    def accuracy_function(self, prbs, labels):
        
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean((tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32)))
        
        return accuracy

def train(model, train_data):
    num_epochs = 0
    while num_epochs < params.max_epochs:
        print('=========================EPOCH ', num_epochs, "==========================\n")
        print('=========================Total No. of Batch ', int(len(train_data[0])/params.batch_size), "==========================\n")
        # Loop through all train_data in batches
        start_index = 0
        while (start_index + params.batch_size) < len(train_data[0]):
            sources, targets, speakers, addressees = data.read_batch(train_data, start_index, mode='train')
            with tf.GradientTape() as tape:
                prbs = model.call(sources, targets[:, :-1])

                loss = model.loss_function(prbs, targets[:,1:])
            
            print('-----------batch ', int(start_index/params.batch_size), ": loss = ", loss, " ---------------\n")
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            start_index += params.batch_size
			# Increment for next epoch
            #
        num_epochs += 1

def test(model, test_data):
    start_index = 0
    total_acc = 0
    loss_list = []

    while (start_index + params.batch_size) < len(test_data[0]):

        sources, targets, speakers, addressees = data.read_batch(test_data, start_index, mode='train')
        prbs = model.call(sources, targets[:, :-1])
        loss = model.loss_function(prbs, targets[:,1:])
        print('-----------batch ', int(start_index/params.batch_size), ": loss = ", loss, " ---------------\n")
        acc = model.accuracy_function(prbs, targets[:,1:])
        print('-----------batch ', int(start_index/params.batch_size), ": acc = ", acc, " ---------------\n")
        batch_total = np.count_nonzero(targets)
        print("batch_total_words: ", batch_total)
        total_acc = total_acc + acc * batch_total
        start_index += params.batch_size
    
    print('=========================Testing Accuracy ', total_acc/np.count_nonzero(test_data[3][0:start_index, :]), "==========================\n")

    show_example(prbs, targets[:,1:])
    l = tf.reduce_mean(loss_list)
    perplexity = tf.math.exp(l)
    print('=========================Perplexity ', perplexity, "==========================\n")

def show_example(probs, labels):
    
    # labels = tf.transpose(labels)
    decoded_vocab_ids = tf.argmax(input=probs, axis=2) 
    decoded_vocab_ids = tf.transpose(decoded_vocab_ids) # shape = (batch_size, sentence_max_length-1)
    print(decoded_vocab_ids.shape)

    for row in range(params.batch_size - 5, params.batch_size, 1):
        sentence = []

        for col in range(0, tf.shape(decoded_vocab_ids)[1], 1):
            print(row, col)
            sentence.append(list(data.vocab_dict.keys())[list(data.vocab_dict.values()).index(decoded_vocab_ids[row][col])])
        
        print(' '.join(word for word in sentence))
        print(labels[row],'\n')




if __name__ == '__main__':

    params = baseline_params()
    data = Data(params)
    print('baseline.py: created params and data')

    friends_data = data.friends_tsv(num_seasons=10)
    data_dict = data.cleanup_and_build_dict(friends_data)
    num_characters = data.num_characters
    num_vocab = len(list(data.vocab_dict.keys()))

    print('num_characters = ', num_characters)
    print('num_vocab = ', num_vocab)
    train_data, test_data = data.train_test_split(data_dict, p_split=0.9) # Friends: num_train = 45416
    model = baseline_model(num_vocab, params)
    print('baseline.py: created baseline_model')
    train(model, train_data)
    print('baseline.py: finished training baseline_model')
    test(model, test_data)
    print('baseline.py: finished testing baseline_model')

