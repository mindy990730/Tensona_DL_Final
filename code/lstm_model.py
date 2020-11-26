from preprocess import Data
import numpy as np
import pickle
import linecache
import math
import tensorflow as tf

class lstm_source(tf.keras.Model):
    def __init___(self, params):
        hidden_sz = params.hidden_sz
        self.lstm_s_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=params.dropout, return_state=True, return_sequences=True)
        # self.dropout_1 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_s_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=params.dropout, return_state=True, return_sequences=True)
        # self.dropout_2 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_s_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=params.dropout, return_state=True, return_sequences=True)
        # self.dropout_3 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_s_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(hidden_sz)
    
    def call(self, input, initial_state):
        """

        Forward pass of source LSTM.

        :param input: source embedding input, shape = (batch_size, sentence_max_length, embed_size)
        :param initial_state: hidden state from previous executions of call

        :return output: output of 4 layers of LSTM, shape = (batch_size, sentence_max_length, hidden_sz)
        :       initial_state: the last hidden state, to be used as initial state of lstm_target
        """
        output, initial_state, _ = self.lstm_s_1(input, initial_state=initial_state)
        # output = self.dropout_1(output)
        output, initial_state, _ = self.lstm_s_2(output, initial_state=initial_state)
        # output = self.dropout_2(output)
        output, initial_state, _ = self.lstm_s_3(output, initial_state=initial_state)
        # output = self.dropout_3(output)
        output, initial_state, _ = self.lstm_s_4(output, initial_state=initial_state)
        output = self.dense(output)
        return output, initial_state


class lstm_target(tf.keras.Model):
    def __init___(self, params, num_vocab, num_characters, mode='SPEAKER'):
        hidden_sz = params.hidden_sz
        self.persona_embedding = tf.random.normal([num_characters, params.embed_size], stddev=.1, dtype=tf.float32)
        self.lstm_t_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_1 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_t_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_2 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_t_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_3 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_t_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(num_vocab)
        self.mode = mode

    def call(self, encoded_input, initial_state, word_embedding, speaker, addressee):
        """

        Forward pass of target LSTM that calculates the probablity of next words. 

        :param encoded_input: output of lstm_source, shape = (batch_size, sentence_max_length, hidden_sz)
        :param initial_state: last hidden_state of lstm_source or previous calls of lstm_target
        :param word_embedding: target embedding input, shape = (batch_size, 1, embed_size)
        :param speaker: list of speakers in this batch, shape = (batch_size)
        :param addressee: list of addressee in this batch, shape = (batch_size)

        :return probs: probilities of vocab for this batch of words, shape = (batch_size, num_vocab)
        :       initial_state: the last hidden state of this call; to be used in the next call
        """
        speaker_embed = tf.nn.embedding_lookup(self.persona_embedding, speaker) # shape = (batch_size, embed_size)
        # lstm_t_input = tf.concat(encoded_input, word_embedding)
        word_embedding = tf.squeeze(word_embedding)
        lstm_t_input = tf.concat(word_embedding, speaker_embed, axis=1) # shape = (batch_size, 2*embed_size)
        # lstm_t_input = encoded_state + target_embedding + speaker embedding
        if self.mode=='SPEAKER_ADDRESSEE':
            addressee_embed = tf.nn.embedding_lookup(self.persona_embedding, addressee) # shape = (batch_size, embed_size)
            lstm_t_input = tf.concat(lstm_t_input, addressee_embed, axis=1) # shape = (batch_size, 3*embed_size)

        output, initial_state, _ = self.lstm_s_1(lstm_t_input, initial_state=initial_state)
        output = self.dropout_1(output)
        output, initial_state, _ = self.lstm_s_2(output, initial_state=initial_state)
        output = self.dropout_2(output)
        output, initial_state, _ = self.lstm_s_3(output, initial_state=initial_state)
        output = self.dropout_3(output)
        output, initial_state, _ = self.lstm_s_4(output, initial_state=initial_state)
        probs = self.dense(output, activation='softmax')

        return probs, initial_state


class MyLstm(tf.keras.Model):
    def __init___(self, params, num_vocab, num_characters):
        self.encoder = lstm_source(params)
        self.decoder = lstm_target(params, num_vocab, num_characters, mode='SPEAKER')
        self.source_embedding = tf.random.normal([num_vocab, params.embed_size], stddev=.1, dtype=tf.float32)
        self.target_embedding = tf.random.normal([num_vocab, params.embed_size], stddev=.1, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = params.lr_rate)
        
    def call(self, batched_source, batched_target, speaker_list, addressee_list):
        """

        Runs the model on one batch of source & target inputs

        :param batched_source: 2-D array of size (batch_size, sentence_max_length) that contains the batched, tokenized source scripts 
        :param batched_target: 2-D array of size (batch_size, sentence_max_length) that contains the batched, tokenized target scripts 
        :param speaker_list: 1-D array of size (batch_size) that contains speaker ids
        :param addressee_list: 1-D array of size (batch_size) that contains addressee ids

        :return loss: a tensor that contains the loss of this batch
        :       probs_list: a 2-D tensor that contains probabilities calculated for each column of 
                            words in target, shape = (sentence_max_length-1, batch_size, num_vocab)

        """
        source_ebd = tf.nn.embedding_lookup(self.source_embedding, batched_source) # shape = (batch_size, sentence_max_length, embed_size)
        # run LSTM encoder on the source sentence
        encoded_output, initial_state = self.encoder(source_ebd, initial_state=None)
        losses = []
        probs_list = []
        # Going horizontally by columns and predict one word each step; compare loss with target next word
        for i in range(tf.shape(batched_target)[1]-1):
            target_ebd = tf.nn.embedding_lookup(self.target_embedding, batched_target[:, i]) # shape = (batch_size, 1, embed_size)
            probs, initial_state = self.decoder(encoded_output, initial_state, target_ebd, speaker_list, addressee_list)
            labels = tf.squeeze(batched_target[:, i+1]) # shape = (batch_size,)
            l = self.loss_function(probs, labels)
            losses.append(l)
            probs_list.append(probs)
        # probs_list shape = [(num_col of batched_target -1), batch_size, num_vocab (probs of each vocab)]
        # labels shape = [(num_col of batched_target -1), batch_size] (transpose of batched_target except the first column of batched_target)
        loss = tf.reduce_sum(losses)
        probs_list = tf.convert_to_tensor(probs_list)
        return loss, probs_list

    def loss_function(self, probs, labels):
        """
		Computes the loss of this columns of words
		:param probs:  float tensor, word prediction probabilities (batch_size, num_vocab)
		:param labels:  integer tensor, word prediction labels (batch_size,)
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
        l = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        l = tf.math.reduce_sum(l)
        return l

    def accuracy_function(self, probs, labels):
        """
		Computes the batch accuracy
		:param probs:  a 2-D tensor that contains probabilities calculated for each column of words
                        in target, shape = (sentence_max_length-1, batch_size, num_vocab)
		:param labels: prediction of next word in target, shape = (sentence_max_length-1, batch_size)
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
        # Hardcoded, #TODO: Try beam search
        decoded_vocabs = tf.argmax(input=probs, axis=2)
        accuracy = tf.reduce_mean((tf.cast(tf.equal(decoded_vocabs, labels), dtype=tf.float32)))
        return accuracy







