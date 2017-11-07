import numpy as np

class HopfieldNetwork(object):
	def __init__(self, size):
		# we initialize the weights of a hopfield network
		self.weights = np.array([[0] * size for i in range(size)])

	def queryPattern(self, inputArray):
		# I convert the input array into a numpy array to allow numpy operations on the array
		inputArray = np.array(inputArray)


	def addPattern(self, inputArray):
		# I convert the input array into a numpy array to allow numpy operations to be done on the matrix.
		inputArray = np.array(inputArray)

		# now we take the binary inputArray and convert it to bipolar values
		inputArray = binaryToBipolar(inputArray)

		# A dot product needs to be done on the trapose of the inputArray and the inputArray,
		# so n*m dimensions times m*n dimensions will result in a matrix of n dimensions
		inputArrayTranspose = np.transpose(inputArray)

		# I dot product the tranpose of the input and the input
		dottedInputs = np.dot(inputArrayTranspose, inputArray)

		# Now we need to fill all the values in the diagonal to 0.
		contributionMatrix = np.fill_diagonal(dottedInputs, 0)

		# Now I change the weight by adding the contribution matrix with the weights of the Hopfield Network
		self.weights += contributionMatrix

	# Converts a binary values to bipolar all 0's in an array will turn to -1.
	def binaryToBipolar(self, x):
		# Vectorize the function.
		# I create a lambda and vectorize it which maps the function across matrix
		# the function gets returned and invoked and the return value of the invoked function is returned.
		return np.vectorize(lambda x: (2 * x) - 1)(x)

	# Converts bipolar values to binary all -1's in an array will turn to 0.
	def bipolarToBinary(self, x):
		# Vectorize the function.
		# I create a lambda and vectorize it which maps the function across matrix
		# the function gets returned and invoked and the return value of the invoked function is returned.
		return np.vectorize(lambda x: (x + 1) / 2)(x)

def main():
	pass


if __name__ == "__main__":
	main()