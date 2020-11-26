from preprocess import Data
from os import path
from io import open
import string
import numpy as np
import pickle
import linecache
import math
import tensorflow as tf

class lstm_source(tf.keras.Model):
    def __init___(self, params):
        hidden_sz = params.hidden_sz
        self.lstm_s_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_1 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_s_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_2 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_s_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_3 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_s_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(hidden_sz)
    
    def call(self, input, initial_state):
        output, initial_state, _ = self.lstm_s_1(input, initial_state=initial_state)
        output = self.dropout_1(output)
        output, initial_state, _ = self.lstm_s_2(output, initial_state=initial_state)
        output = self.dropout_2(output)
        output, initial_state, _ = self.lstm_s_3(output, initial_state=initial_state)
        output = self.dropout_3(output)
        output, initial_state, cell_state = self.lstm_s_4(output, initial_state=initial_state)
        output = self.dense(output)
        return output, initial_state, cell_state


class lstm_target(tf.keras.Model):
    def __init___(self, params, mode='SPEAKER'):
        hidden_sz = params.hidden_sz
        self.persona_embedding = tf.random.normal([params.num_characters, params.embed_size], stddev=.1, dtype=tf.float32)
        self.lstm_t_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_1 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_t_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_2 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_t_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dropout_3 = tf.keras.layers.Dropout(params.dropout)
        self.lstm_t_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(hidden_sz)
        self.mode = mode

    def call(self, encoded_input, initial_state, word_embedding, speaker, addressee):
        speaker_embed = tf.nn.embedding_lookup(self.persona_embedding, speaker)
        lstm_t_input = tf.concat(encoded_input, word_embedding)
        lstm_t_input = tf.concat(lstm_t_input, speaker_embed)
        # lstm_t_input = encoded_state + target_embedding + speaker embedding
        if self.mode=='SPEAKER_ADDRESSEE':
            addressee_embed = tf.nn.embedding_lookup(self.persona_embedding, addressee)
            lstm_t_input = tf.concat(lstm_t_input, addressee_embed)

        output, initial_state, _ = self.lstm_s_1(lstm_t_input, initial_state=initial_state)
        output = self.dropout_1(output)
        output, initial_state, _ = self.lstm_s_2(output, initial_state=initial_state)
        output = self.dropout_2(output)
        output, initial_state, _ = self.lstm_s_3(output, initial_state=initial_state)
        output = self.dropout_3(output)
        output, initial_state, cell_state = self.lstm_s_4(output, initial_state=initial_state)
        output = self.dense(output, activation='softmax')
        return output, initial_state, cell_state


class lstm(tf.keras.Model):
    def __init___(self, params, num_vocab):
        self.encoder = lstm_source(params)
        self.decoder = lstm_target(params)
        self.source_embedding = tf.random.normal([num_vocab, params.embed_size], stddev=.1, dtype=tf.float32)
        self.target_embedding = tf.random.normal([num_vocab, params.embed_size], stddev=.1, dtype=tf.float32)
        
    def call(self, batched_source, batched_target, speaker_list, addressee_list):
        source_ebd = tf.nn.embedding_lookup(self.source_embedding, batched_source)
        # run LSTM encoder on the source sentence
        encoded_output, initial_state, _ = self.encoder(source_ebd, initial_state=None)
        losses = []
        # predict one word each step; compare loss with target next word
        for i in range(tf.shape(batched_target)[1]-1):
            target_ebd = tf.nn.embedding_lookup(self.target_embedding, batched_target[:, i])
            probs, initial_state, _ = self.decoder(encoded_output, initial_state, target_ebd, speaker_list, addressee_list)
            l = self.loss_function(probs, batched_target[:, i+1])
            losses.append(l)
        loss = tf.reduce_sum(losses)
        return loss

    def loss_function(self, probs, labels):
        l = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        l = tf.math.reduce_sum(l)
        return l







