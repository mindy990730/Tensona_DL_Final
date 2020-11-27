import numpy as np
import tensorflow as tf

class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):		
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask
		self.weight_K = self.add_weight("K",shape=[input_size, output_size],dtype = "float32")
		self.weight_V = self.add_weight("V",shape=[input_size, output_size],dtype = "float32")
		self.weight_Q = self.add_weight("Q",shape=[input_size, output_size],dtype = "float32")
		
	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		K = tf.tensordot(inputs_for_keys, self.weight_K, [[2],[0]])
		V = tf.tensordot(inputs_for_values, self.weight_V, [[2],[0]])
		Q = tf.tensordot(inputs_for_queries, self.weight_Q, [[2],[0]])

		matrix = Attention_Matrix(K, Q, self.use_mask)
		matrix = tf.matmul(matrix,V)

		return matrix

@av.att_mat_func
def Attention_Matrix(K, Q, use_mask):
	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	matrix = tf.matmul(Q, K, transpose_b = True)/np.sqrt(window_size_keys)
	if use_mask == True:
		matrix += atten_mask

	matrix = tf.nn.softmax(matrix)
	
	return matrix

