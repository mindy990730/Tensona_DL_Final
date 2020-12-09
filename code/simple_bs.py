from math import log
import numpy as np
import tensorflow as tf
from numpy.random import randint

sample_data =np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
		[0.2, 0.1, 0.3, 0.5, 0.4],
        [0.4, 0.1, 0.3, 0.2, 0.5],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]
        ])

# encoded_outputs = tf.random.normal([10,20,10], 10, 2, tf.float32)
# init_probs = tf.random.normal([10, 100], 3, 1, tf.float32)
# init_state = tf.random.normal([2,10,20], 6, 1, tf.float32)
# speaker_list = tf.convert_to_tensor(randint(1, 10, 10))
# addressee_list = tf.convert_to_tensor(randint(1, 10, 10))

# batched_source = tf.convert_to_tensor((np.random.random ([10,20]) * 100).astype(int))
# batched_target = tf.convert_to_tensor((np.random.random ([10,20]) * 100).astype(int))


def simple_beam_search(data, k):
	candidiates = [[list(), 0.0]]
	
    # for each step/column
	for col in data.T:
		all_candidates = []

		# expand each current candidate
		for i in range(len(candidiates)):
			seq, score = candidiates[i]
			for row in range(len(col)):
                # calculate individual scores
				candidate = [seq + [row], score - log(col[row])]
				all_candidates.append(candidate)
                
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		candidiates = ordered[:k]
	return candidiates
 
result = simple_beam_search(sample_data, 2)
for r in result:
	print(r)




