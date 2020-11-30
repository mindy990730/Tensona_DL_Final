import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os


# class lstm_decoder(lstm_model):
#     def call(self, source, target,length,speaker_label,addressee_label):
#         s_embed = tf.nn.embedding_lookup(self.source_embedding, source)
#         encoded_output, initial_state = self.encoder(s_embed, initial_state=None)
#         losses = []

			# if self.params.setting == 'beam_search':
			# 	start=self.tembed(targets[:,0])
			# 	return self.beam_search(start,context,h,c,speaker_label,addressee_label)
			# else: 
			# 	start=self.tembed(targets[:,0])
			# 	decoder_output, state = self.decoder(context,state, start,speaker_label,addressee_label)
			# 	predicted_word = self.sample(pred)
			# 	prediction = predicted_word.unsqueeze(1).clone()
			# 	for i in range(1,self.params.max_decoding_length):
			# 		pred,h,c=self.decoder(context,h,c,self.tembed(predicted_word),speaker_label,addressee_label)
			# 		predicted_word = self.sample(pred)
			# 		prediction = torch.cat((prediction,predicted_word.unsqueeze(1).clone()),1)
			# 		if (prediction==self.EOT).any(1).all():
			# 			break
			# 	return prediction

        # pass

    

class decoder_model(tf.keras.Model):
    def __init__(self, beam_size, max_decoding_length):
        super(decoder_model, self).__init__()

        # self.data = data
        # friends_data_dict = self.data.friends_tsv(num_seasons=10)
        # self.friends_data = self.data.cleanup_and_build_dict(friends_data_dict)
        # self.num_vocab = len(list(self.data.vocab_dict.keys())) # 15105
        # self.num_characters = len(list(self.data.character_dict.keys()))
        # self.embed_size = 256
        # self.batch_size = 64
        # self.hidden_sz = 512
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # self.lstm = lstm_model(self.params, self.num_vocab, self.num_characters)
        # self.dense = tf.keras.layers.Dense(self.hidden_size, activation='softmax')
        # self.attention = Attention
        self.beam_size = beam_size #200 in the paper
        self.max_decoding_length = max_decoding_length #20 in the paper
        
    # def call(self, input, hidden, encoder_outputs):
    #     """
    #     params: 
    #     input           -> (1 x Batch Size)
    #     speakers        -> (1 x Batch Size, Addressees of inputs to Encoder)
    #     hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
    #     encoder_outputs -> (Max Sentence Length-1, Batch Size, Hidden Size)

    #     return:
    #     """

    #     features = self.embedding(input)
    #     output, _,_ = self.lstm(features)
    #     output = self.dense(output)
    #     return output

    def beam_search(self, inputs, stop_token, max_decoding_length,beam_size):
        """
        Performs beam search to produce the N-best lists

        :param inputs:  a 3-D tensor that contains probabilities calculated for each column of words
                        in target, shape = (sentence_max_length-1, batch_size, num_vocab)
        :param stop_token: a token that signals the end of a single hypothesis
        :param max_decoding_length: maximumum length of generated candidates 
        :param beam_size: the scope of next-word candidates to search for 
        :return: N-best list
        """
        
        #convert inputs to hypothesis
        unfinished = []
        steps = 0
        best_candidates = []

        # At each time step, examine all B × B possible next-word candidates
        while steps <= self.max_decoding_length and len(best_candidates) < self._beam_size:
           curr_candidates = []
        #    hyp = ？
        #   if hyp ends in stop_token:
        #         best_candidates.append(hyp)
        #   else:
        #       unfinished.append(hyp)

        # preserve the top-B unfinished hypotheses: 
        # next_candidates, word_indices = tf.math.top_k(unfinished[something], k=beam_size)

        # rerank the generated N-best list using a scoring function that linearly combines 
        # a length penalty and the log likelihood of the source given the target

        # length_penalty = tf.div((5. + tf.to_float(sequence_lengths))**penalty_factor, (5. + 1.)
        #             **penalty_factor)
        # log_likelihood = ?
        # score = log_probs / length_penality
        # rank curr_candidates by score

        return best_candidates



## AN EXAMPLE USING TF BUILT-IN FUNCTIONS:

# my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
#           cell=cell,
#           embedding=embedding_decoder,
#           start_tokens=start_tokens,
#           end_token=end_token,
#           initial_state=decoder_initial_state,
#           beam_width=200,
#           output_layer=output_layer)
        
# outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
#         my_decoder,
#         maximum_iterations=maximum_iterations,
#         output_time_major=time_major,
#         swap_memory=True,
#         scope=decoder_scope)
# sample_id = outputs.predicted_ids