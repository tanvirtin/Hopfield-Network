from sklearn import datasets
from sklearn.datasets import fetch_mldata
import numpy as np

def filteredMnist():
	print("Fetching the dataset...")
	digits = fetch_mldata('MNIST original', data_home=".\\")
	
	# contains many 2 dimensional array of pixel values which represents digits
	images = digits.data

	# contains labels for the pixel values
	labels = digits.target

	images = images.astype(np.float32)

	indexOnes = []

	indexFives = []

	for i in range(len(labels)):
		if labels[i] == 1:
			indexOnes.append(i)
		# first condition is not met we try the second condition
		elif labels[i] == 5:
			indexFives.append(i)

	ones = [images[indexOnes[i]] for i in range(len(indexOnes))]

	fives = [images[indexFives[i]] for i in range(len(indexFives))]

	print("Ones and fives filtered...")

	return (np.array(ones), np.array(fives))
