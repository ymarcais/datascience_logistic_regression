import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pair_plot import Pairplot_graph
import csv
import time

@dataclass
class Logreg_train:
	dataframe : np.ndarray = None
	ema_beta = 0.9
    
	#get data
	def get_data_(self, path):
		pg = Pairplot_graph()
		self.dataframe = pg.data_(path)
			
	#list unique Hogwarts Houses
	def get_houses_list(self, path):
		houses_list = self.dataframe['Hogwarts House'].unique()
		return houses_list
	
	# create output 'y' houses '1/0' one vs all
	def y_houses(self, path):
		houses_list = self.get_houses_list(path)
		for house in houses_list:
			column_name = 'y_' + house
			self.dataframe[column_name] = (self.dataframe['Hogwarts House'] == house).astype(int)
	
	#get X_train and reshape
	def X_train(self):
		X_train = self.dataframe.iloc[:, 1:14].values
		X_train = X_train.T
		return X_train
	
	#get Y_train and reshape
	def Y_train(self, X_train):
		Y_train = self.dataframe.iloc[:, 14:18].values
		return Y_train

	# key function to help separate into 2 parts 1 / 0
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	
	# use the derivatives of cost function to reduce the cost by iteration
	def gradient_descent(self, Y, A, X, m, cost_list):
		cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
		dW = (1/m) * np.dot(A - Y, X.T)
		dB = (1/m) * np.sum(A - Y)
		cost_list.append(cost)
		return cost_list, dW, dB

	# model regression model with loss gradient descent
	def models(self, X, Y, learning_rate, iterations):
		cost_list =[]
		m = X.shape[1]
		n = X.shape[0]
		W = np.zeros((n, 4))
		B = 0

		Y = Y.T
		for i in range(iterations):
			Z = np.dot(W.T, X) + B
			A = self.sigmoid(Z)
			
			cost_list, dW, dB = self.gradient_descent(Y, A, X, m, cost_list)

			W -= learning_rate * dW.T
			B -= learning_rate * dB
		return W, B, cost_list
	
	# Aleternative to "models()" with stochastic batch and Exponential Moving Average
	# But not very relevant in time with those data
	def mini_batch_model(self, X, Y, learning_rate, iterations):
		cost_list =[]
		m = X.shape[1]
		n = X.shape[0]
		ema_W = np.zeros((n, 4))
		ema_B = 0
		batch_size = 200
		start_row = 0
		end_row = m - m % 200
		j = 0
		W = np.zeros((n, 4))
		B = 0

		Y = Y.T

		for start_row in range(0, end_row, batch_size):
			batch = X[:, start_row:start_row + batch_size]
			Y_batch = Y[:, start_row:start_row + batch_size]

			for i in range(iterations):
				Z = np.dot(W.T, batch) + B
				A = self.sigmoid(Z)

				cost_list, dW, dB = self.gradient_descent(Y_batch, A, batch, m, cost_list)
				W -= learning_rate * dW.T
				B -= learning_rate * dB
				j += 1
				''' Exponential Moving Average (EMA)
				ema_W = self.ema_beta * ema_W + (1 - self.ema_beta) * dW.T
				ema_B = self.ema_beta * ema_B + (1 - self.ema_beta) * dB

				W_update = learning_rate * ema_W
				W -= W_update
				
				B_update = learning_rate * ema_B
				B -= B_update'''
		return W, B, cost_list, j
	
	#Store W and B in CSV file
	def save_weights(self, W, B):
		file_path = 'datasets/weights.csv'
		with open(file_path, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['Weight']* W.shape[1] + ['Bias'])
			for i in range(W.shape[0]):
				writer.writerow([W[i][j] for j in range(W.shape[1])] + [B])
		
	#Calculate the accuracy of the models
	def accuracy(self, X, Y, W, B):
		Z = np.dot(W.T, X) + B
		A = self.sigmoid(Z)
		A = A > 0.5

		A = np.array(A, dtype= 'int64')
		Y = Y.T
		acc = (1 -np.sum(np.absolute(A - Y))/ Y.shape[1])*100
		acc = f"{acc:.2f}"
		print("Accuracy : ", acc, "%")

	#Plot charts
	def plot_cost_loss(self, j, cost_list):
		title_lines = [
		'Multi Class Logistic Regression',
		'Mini Batch Stochastic Gradient Descent',
		'With Exponential Moving Averga (EMA)',
		]
		title = '\n'.join(title_lines)

		plt.title(title, color='blue')
		plt.plot(np.arange(j), cost_list, color='red')
		plt.show()

def	main():
	start = time.time()
	path = "datasets/dataset_train.csv"
	iterations = 100
	learning_rate = 0.01
	lt = Logreg_train()
	lt.get_data_(path)
	lt.y_houses(path)
	X_train = lt.X_train()
	Y_train = lt.Y_train(X_train)
	#W, B, cost_list = lt.models(X_train, Y_train, learning_rate, iterations)
	W, B, cost_list, j = lt.mini_batch_model(X_train, Y_train, learning_rate=learning_rate, iterations=iterations)
	lt.save_weights(W, B)
	lt.accuracy(X_train, Y_train, W, B)
	#lt.plot_cost_loss(iterations, cost_list)
	lt.plot_cost_loss(j, cost_list)
	end = time.time()
	print("time:", end - start)

if __name__ == "__main__":
    main()