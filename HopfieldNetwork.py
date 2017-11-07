import numpy as np

class HopfieldNetwork(object):
	def __init__(self, size):
		# we initialize the weights of a hopfield network
		self.weights = np.array([[0] * size for i in range(size)])
		self.size = size

	def queryPattern(self, inputArray):
		lengthOfInputs = len(inputArray)

		# I convert the input array into a numpy array to allow numpy operations on the array
		inputArray = np.array([inputArray])

		# Now we take the bipolar inputArray and convert it to binary.
		inputArray = self.bipolarToBinary(inputArray)

		# Now what we have to do is take this inputArray and dot product it with
		# every single column in of the weight array, and if the value if greater
		# than 0 it means that the neuron has firetd

		# Now the each index of the weightTranposed array will give you a column
		# of the weight matrix.
		weightTransposed = np.transpose(self.weights)

		# holds the prediction made by the HopfieldNetwork
		outputArray = []

		# we loop inputArray amount of times as we are going to 
		for i in range(self.size):
			# I transpose the column and dot product it with the inputArray
			# to obtain one single value, if this value is positive we add 1 to
			# the output array if its a negative we add 0 to the outputArray

			# When transposing I have to make sure that we are transposing a 2-D
			# array. 
			transposedColumn = np.transpose(np.array([weightTransposed[i]]))

			outcome = np.dot(inputArray, transposedColumn)

			if outcome > 0:
				outputArray.append(1)
			else:
				outputArray.append(0)

		return outputArray


	def addPattern(self, inputArray):
		# I convert the input array into a numpy array to allow numpy operations to be done on the matrix.
		inputArray = np.array([inputArray])

		# now we take the binary inputArray and convert it to bipolar values
		inputArray = self.binaryToBipolar(inputArray)

		# A dot product needs to be done on the trapose of the inputArray and the inputArray,
		# so n*m dimensions times m*n dimensions will result in a matrix of n dimensions
		inputArrayTranspose = np.transpose(inputArray)

		# I dot product the tranpose of the input and the input
		dottedInputs = np.dot(inputArrayTranspose, inputArray)

		# Now we need to fill all the values in the diagonal to 0.
		np.fill_diagonal(dottedInputs, 0)

		# Now I change the weight by adding the contribution matrix with the weights of the Hopfield Network
		self.weights += dottedInputs

	# Converts a binary values to bipolar all 0's in an array will turn to -1.
	def binaryToBipolar(self, x):
		# Vectorize the function.
		# I create a lambda and vectorize it which maps the function across matrix
		# the function gets returned and invoked and the return value of the invoked function is returned.
		return np.vectorize(lambda x: (2 * x) - 1)(x)

	# Converts bipolar values to binary all -1's in an array will turn to 0.
	def bipolarToBinary(self, x):
		def vectorizedFunc(a):
			if a == 0:
				return -1
			else:
				return a
		# Vectorize the function.
		# I create a lambda and vectorize it which maps the function across matrix
		# the function gets returned and invoked and the return value of the invoked function is returned.
		return np.vectorize(vectorizedFunc)(x)

def main():
		
	hnn = HopfieldNetwork(4)

	data = [0, 1, 0, 1]

	hnn.addPattern(data)

	dataTwo = [1, 1, 1, 1]

	hnn.addPattern(dataTwo)

	hnn.addPattern([1, 1, 1, 0])

	print(hnn.queryPattern(data))

	print(hnn.queryPattern([1, 1, 1, 1]))




if __name__ == "__main__":
	main()