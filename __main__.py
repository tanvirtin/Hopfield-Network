from hopfieldNetwork import hopfieldNetwork, queryPattern, binaryToBipolar, bipolarToBinary
from filteredMnist import filteredMnist
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

def findAccuracy(y, prediction):
	y = bipolarToBinary(y)
	prediction = bipolarToBinary(prediction)

	accuracy = 0

	# loop over y images
	for i in range(len(y)):
		# loop over each pixel in y image
		for j in range(len(y[i])):
			# if the pixel values match I add it to accuracy
			if y[i][j] == prediction[i][j]:
				accuracy += 1

	accuracy /= len(y)

	accuracy /= len(y[0])

	return accuracy * 100

def trainBatches(ones, fives, numTrainData):
	train = []

	for i in range(numTrainData):
		train.append(ones[i])
		train.append(fives[i])

	train = np.array(train)

	test = np.array([ones[random.randint(0, len(ones) - 1)], fives[random.randint(0, len(fives) - 1)]])

	train = binaryToBipolar(train)

	reshapeDim = int(round(np.sqrt(len(train[0]))))

	weights = hopfieldNetwork(train)

	results = queryPattern(weights, test) 

	return findAccuracy(test, results)


def main():
	ones, fives = filteredMnist()

	batchSize = 100

	for i in tqdm(range(0, 10000, 100)):
		print("Accuracy is {}% for {} images of ones and fives".format(trainBatches(ones, fives, batchSize), i * 2))




if __name__ == "__main__":
	main()