
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))


class LogisticRegression():

	def __init__(self, lr =0.1, iterations = 10000):
		self.lr = lr
		self.iterations = iterations
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		#self.weights = np.random.rand(n_features)-0.5
		self.bias = 0
		#self.bias = np.random.rand(1)-0.5

		for _ in range(self.iterations):
			linear_predictions = np.dot(X,self.weights) + self.bias
			predictions = sigmoid(linear_predictions)
			#print(predictions)
			dw = (1/n_samples)* np.dot(X.T, (predictions-y))
			db = (1/n_samples)* np.sum(predictions-y)

			self.weights = self.weights - self.lr * dw
			self.bias = self.bias - self.lr * db
		#print(self.weights)

	def predict(self, X):
		linear_predictions = np.dot(X,self.weights) + self.bias
		y_predictions = sigmoid(linear_predictions)
		return y_predictions
