from math import log
import numpy as np
import tensorflow as tf

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
 
def simple_beam_search(data, k):
	candidiates = [[list(), 0.0]]
	
    # for each step
	for row in data:
		all_candidates = []

		# expand each current candidate
		for i in range(len(candidiates)):
			seq, score = candidiates[i]
			for j in range(len(row)):
                # calculate individual scores
				candidate = [seq + [j], score - log(row[j])]
				all_candidates.append(candidate)
                
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		candidiates = ordered[:k]
	return candidiates
 
# result = simple_beam_search(sample_data, 2)
# for r in result:
# 	print(r)



sample = sample_data[:2]
print("old = ",sample)
sample = tf.expand_dims(sample_data[:2], axis = 1)
print("new = ",sample)