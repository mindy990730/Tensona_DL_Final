import numpy as np
import tensorflow as tf
import numpy as np

class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):		
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask

		# Initialize the weight matrices for K, V, and Q.
		# They should be able to multiply an input_size vector to produce an output_size vector 
		self.K = self.add_weight(shape=(input_size, output_size), initializer="random_normal", trainable=True)
		self.V = self.add_weight(shape=(input_size, output_size), initializer="random_normal", trainable=True)
		self.Q = self.add_weight(shape=(input_size, output_size), initializer="random_normal", trainable=True)

	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

		"""

		This functions runs a single attention head.
        # TODO:
		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""

		# - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this. 
		# - Call Attention_Matrix with the keys and queries, and with self.use_mask.
		# - Apply the attention matrix to the values

		K = tf.tensordot(inputs_for_keys, self.K, 1)
		V = tf.tensordot(inputs_for_values, self.V, 1)
		Q = tf.tensordot(inputs_for_queries, self.Q, 1)

        K_T = tf.transpose(K, perm=[0,2,1])
        Q_times_K_T = tf.matmul(Q, K_T)
        # if use_mask:
        #     Q_times_K_T += atten_mask
        scale = tf.math.sqrt(tf.cast(K.get_shape()[2], tf.float32))
        scaled_Q_K = tf.math.divide(Q_times_K_T, scale)
        result = tf.nn.softmax(scaled_Q_K)
		Z = tf.matmul(result, V)
		return Z
