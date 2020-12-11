import tensorflow as tf
import numpy as np
from tensorflow.keras import Model


class test_target(tf.keras.Model):
    def __init__(self, params, num_vocab, num_characters, is_speaker):

        super(test_target, self).__init__()
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


    def call(self, encoded_output, initial_state, word_embedding, speaker, addressee, is_first_word):
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
        # if os.path.isdir('../saved_weights') and is_test:
		# 		self.model.encoder.load_weights('../saved_weights/en_weights.tf')
        # self.load_weights('../saved_weights/de_weights.tf')
        # print('Loaded existing weights.')

        # Concatenate embeddings 
        if is_first_word:
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
        else: #encoded output = prebvious word embedding (batch_sz, 1, emb_sz)
            reshaped_encoder_output = tf.expand_dims(encoded_output, axis=1)
        # Now reshaped_encoder_output shape = (batch_size, 1, sentence_max_length * embed_size) or (batch_sz, 1, emb_sz)
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
