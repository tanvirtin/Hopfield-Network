import numpy as np

# takes in patterns that the neural network will learn
def hopfieldNetwork(patterns):
	# I make a weight array with length of a pattern * length of a pattern
	weights = np.zeros((len(patterns[0]), len(patterns[0])))

	# take the mean of all the patterns
	mean = np.mean(patterns)

	# I deduct mean from patterns array
	patterns -= mean

	for pattern in patterns:
		weights += np.outer(pattern, pattern) / len(patterns)

	return weights

def bipolarToBinary(weights):
	weights[weights < 0] = 0
	weights[weights > 0] = 1
	weights[weights == 0] = 0

	return weights

def binaryToBipolar(weights):
	weights[weights < 0] = -1
	weights[weights > 0] = 1
	weights[weights == 0] = -1

	return weights

def queryPattern(weights, testPatterns, iteration = 100):
	# we need to fill the diagonals with 0 first
	np.fill_diagonal(weights, 0)

	testPatterns = binaryToBipolar(testPatterns)

	results = []

	for pattern in testPatterns:
		
		newPattern = weights.dot(pattern) - iteration

		newPattern = bipolarToBinary(newPattern)

		results.append(newPattern)

	return np.array(results)
