from preprocess import Data
# from attention import *
import numpy as np
import pickle
import linecache
import math
import tensorflow as tf
import os
# from decode import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class lstm_source(tf.keras.Model):
    def __init__(self, params):

        super(lstm_source, self).__init__()

        # self.self_attention = Atten_Head(params.embed_size,params.embed_size,use_mask=False)
        # self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        hidden_sz = params.hidden_sz
        self.lstm_s_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=params.dropout, return_state=True, return_sequences=True)
        self.lstm_s_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=params.dropout, return_state=True, return_sequences=True)
        self.lstm_s_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=params.dropout, return_state=True, return_sequences=True)
        self.lstm_s_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense_1 = tf.keras.layers.Dense(hidden_sz, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(params.embed_size, activation=None)
    
    def call(self, inputs, initial_state):
        """

        Forward pass of source LSTM.

        :param input: source embedding input, shape = (batch_size, sentence_max_length, embed_size)
        :param initial_state: hidden state from previous executions of call

        :return output: output of 4 layers of LSTM, shape = (batch_size, sentence_max_length, embed_size)
        :       initial_state: the last hidden state, to be used as initial state of lstm_target
        """
        # TODO: create self attention

        output, h, c = self.lstm_s_1(inputs, initial_state=initial_state)
        # print('\n\nlstm_s_1 output = ', output)
        initial_state = (h,c)
        output, h, c = self.lstm_s_2(output, initial_state=initial_state)
        # print('\n\nlstm_s_2 output = ', output)
        initial_state = (h,c)
        output, h, c = self.lstm_s_3(output, initial_state=initial_state)
        # print('\n\nlstm_s_3 output = ', output)
        initial_state = (h,c)
        output, h, c = self.lstm_s_4(output, initial_state=initial_state)
        # print('\n\nlstm_s_4 output = ', output)
        initial_state = (h,c)
        output = self.dense_1(output)
        # print('\n\nsource dense_2 output = ', output)

        output = self.dense_2(output)
        return output, initial_state


class lstm_target(tf.keras.Model):
    def __init__(self, params, num_vocab, num_characters, is_speaker):

        super(lstm_target, self).__init__()
        self.is_speaker = is_speaker
        self.params = params
        hidden_sz = params.hidden_sz
        self.persona_embedding = tf.random.normal([num_characters, self.params.embed_size], stddev=.1, dtype=tf.float32)
        self.lstm_t_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=self.params.dropout, return_state=True, return_sequences=True)
        self.lstm_t_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=self.params.dropout, return_state=True, return_sequences=True)
        self.lstm_t_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=self.params.dropout, return_state=True, return_sequences=True)
        self.lstm_t_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(num_vocab, activation='softmax')

        if not self.is_speaker:
            self.speaker_linear = tf.keras.layers.Dense(self.params.embed_size)
            self.addressee_linear = tf.keras.layers.Dense(self.params.embed_size)
            self.s_a_linear = tf.keras.layers.Dense(self.params.embed_size, activation='tanh')


    def call(self, encoded_output, initial_state, word_embedding, speaker, addressee):
        """

        Forward pass of target LSTM that calculates the probablity of next words. 

        :param encoded_output: output of lstm_source, shape = (batch_size, sentence_max_length, embed_size)
        
        :param initial_state: last hidden_state of lstm_source or previous calls of lstm_target
        :param word_embedding: target embedding input, shape = (batch_size, 1, embed_size)
        :param speaker: list of speakers in this batch, shape = (batch_size)
        :param addressee: list of addressee in this batch, shape = (batch_size)
        :param is_speaker: True if speaker model, False if speaker-addressee model

        :return probs: probilities of vocab for this batch of words, shape = (batch_size, num_vocab)
        :       initial_state: the last hidden state of this call; to be used in the next call
        """
        # Concatenate embeddings 
        for i in range(self.params.batch_size): # in each sentence of this bacth
            word_embeddings_in_s = tf.slice(encoded_output, [i, 0, 0], [1, 1, self.params.embed_size]) # shape = (1, 1, embed_size)
            for s in range(1, tf.shape(encoded_output)[1], 1):
                w_e = tf.slice(encoded_output, [i, s, 0], [1, 1, self.params.embed_size])
                word_embeddings_in_s = tf.concat((word_embeddings_in_s, w_e), axis=2)
            # Now word_embeddings_in_s shape = (1, 1, sentence_max_length * embed_size)
            if i == 0:
                reshaped_encoder_output = word_embeddings_in_s
            else:
                reshaped_encoder_output = tf.concat((reshaped_encoder_output, word_embeddings_in_s), axis=0)

        # Now reshaped_encoder_output shape = (batch_size, 1, sentence_max_length * embed_size)
        word_embedding = tf.expand_dims(word_embedding, axis=1)
        lstm_t_input = tf.concat([reshaped_encoder_output, word_embedding], axis=2) # shape = (batch_size, 1, (1+sentence_max_length) * embed_size)

        speaker_embed = tf.nn.embedding_lookup(self.persona_embedding, speaker) # shape = (batch_size, embed_size)
        speaker_embed = tf.expand_dims(speaker_embed, axis=1) # shape = (batch_size, 1, embed_size)
        
        if self.is_speaker:
            lstm_t_input = tf.concat([lstm_t_input, speaker_embed], axis=2) # shape = (batch_size, 1, (2+sentence_max_length) * embed_size)
        else:
            addressee_embed = tf.nn.embedding_lookup(self.persona_embedding, addressee) # shape = (batch_size, embed_size)
            addressee_embed = tf.expand_dims(addressee_embed, axis=1)  # shape = (batch_size, 1, embed_size)
            
            addressee_embed = self.addressee_linear(addressee_embed) # shape = (batch_size, 1, embed_size)
            speaker_embed = self.speaker_linear(speaker_embed) # shape = (batch_size, 1, embed_size)
            
            combined_embed = tf.concat([speaker_embed, addressee_embed], axis=2) # shape = (batch_size, 1, 2 * embed_size)
            combined_embed = self.s_a_linear(combined_embed) # shape = (batch_size, 1, embed_size)
            lstm_t_input = tf.concat([lstm_t_input, combined_embed], axis=2) # shape = (batch_size, 1, (3+sentence_max_length) * embed_size)
        
        output, h, c = self.lstm_t_1(lstm_t_input, initial_state=initial_state)
        initial_state = (h,c)
        output, h, c = self.lstm_t_2(output, initial_state=initial_state)
        initial_state = (h,c)
        output, h, c = self.lstm_t_3(output, initial_state=initial_state)
        initial_state = (h,c)
        output, h, c = self.lstm_t_4(output, initial_state=initial_state)
        initial_state = (h,c)
        probs = self.dense(output) # shape = (batch_size, 1, num_vocab)
        probs = tf.squeeze(probs)
        return probs, initial_state


class lstm_model(tf.keras.Model):

    def __init__(self, params, num_vocab, num_characters, is_speaker):
        super(lstm_model, self).__init__()
        self.is_speaker = is_speaker
        self.encoder = lstm_source(params)
        self.decoder = lstm_target(params, num_vocab, num_characters, self.is_speaker)
        self.source_embedding = tf.random.normal([num_vocab, params.embed_size], stddev=.1, dtype=tf.float32)
        self.target_embedding = tf.random.normal([num_vocab, params.embed_size], stddev=.1, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = params.lr_rate)
        self.beam_size = 200
        self.sentence_max_length = 20
        self.batch_size = 64
        # print('lstm_model is built!!!!!!!!!!!!!\n\n')

    def call(self, batched_source, batched_target, speaker_list, addressee_list, initial_state):
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
        # print('\nsource embedding = ', source_ebd)
        # run LSTM encoder on the source sentence
        encoded_output, initial_state = self.encoder(source_ebd, initial_state=initial_state)
        losses = []
        probs_list = []
        # Going horizontally by columns and predict one word each step; compare loss with target next word
        for i in range(tf.shape(batched_target)[1]-1):
            target_ebd = tf.nn.embedding_lookup(self.target_embedding, batched_target[:, i]) # shape = (batch_size, 1, embed_size)
            probs, initial_state = self.decoder(encoded_output, initial_state, target_ebd, speaker_list, addressee_list)
            labels = tf.squeeze(batched_target[:, i+1]) # shape = (batch_size,)
            l = self.loss_func(probs, labels)
            # if i%5 ==0:
                # print('loss in the sentence = ', l)
            losses.append(l)
            probs_list.append(probs)
        losses = tf.convert_to_tensor(losses) 
        loss = tf.reduce_sum(losses)
        probs_list = tf.convert_to_tensor(probs_list)
        return loss, probs_list

    def loss_func(self, probs, labels):
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
        :param probs:  a 3-D tensor that contains probabilities calculated for each column of words
                        in target, shape = (sentence_max_length-1, batch_size, num_vocab)
        :param labels: prediction of next word in target, shape = (sentence_max_length-1, batch_size)
        :return: scalar tensor of accuracy of the batch between 0 and 1     
        """
        #TODO: Try beam search
        # Hardcoded
        decoded_vocabs = tf.cast(tf.argmax(input=probs, axis=2), dtype=tf.int64)
        accuracy = tf.reduce_mean((tf.cast(tf.equal(decoded_vocabs, labels), dtype=tf.float32)))
        return accuracy