import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pair_plot import Pairplot_graph
import csv

@dataclass
class Logreg_train:
	dataframe : np.ndarray = None
    
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
		#print("self.dataframe :", self.dataframe.iloc[:, :-4])
		#print("self.dataframe last four columns:", self.dataframe.iloc[:, -4:])

	
	#get X_train and reshape
	def X_train(self):
		X_train = self.dataframe.iloc[:, 1:14].values
		X_train = X_train.T
		#print(X_train)
		return X_train
	
	#get Y_train and reshape
	def Y_train(self, X_train):
		Y_train = self.dataframe.iloc[:, 14:18].values
		#print("Y_train :",Y_train[:10])
		#Y_train = Y_train.reshape(4, X_train.shape[1])
		#Y_train = pd.DataFrame(Y_train)
		#print("Y_train :",Y_train[:10])
		return Y_train

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def models(self, X, Y, learning_rate, iterations):
		cost_list =[]
		m = X.shape[1]
		n = X.shape[0]

		W = np.zeros((n, 4))
		B = 0
		
		x, y = W.shape
		#print("W shape :", x, " X", y)
		x, y = X.shape
		#print("X shape :", x, " X", y)	

		Y = Y.T
		for i in range(iterations):
			Z = np.dot(W.T, X) + B
			#Y = Y.T

			A = self.sigmoid(Z)
			#x, y = A.shape
			#print("A shape :", x, " X", y)
			#x, y = Y.shape
			#print("Y shape :", x, " X", y)

			
			cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
			dW = (1/m) * np.dot(A - Y, X.T)
			dB =  (1/m) * np.sum(A - Y)

			W -= learning_rate * dW.T
			B -= learning_rate * dB

			cost_list.append(cost)

			if(i % (iterations / 10 ) == 0):
				print("cost after :", i, "iterations is : ", cost)
		x, y = Z.shape
		return W, B, cost_list
	
	'''def save_weights(self, W):
		file_path = 'datasets/weights.csv'
		with open(file_path, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['Weight'] * W.shape[1])
			writer.writerows(W)'''
	
	def save_weights(self, W, B):
		file_path = 'datasets/weights.csv'
		with open(file_path, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['Weight']* W.shape[1] + ['Bias'])  # Include headers for Bias as well
			for i in range(W.shape[0]):
				writer.writerow([W[i][j] for j in range(W.shape[1])] + [B])
		
	def accuracy(self, X, Y, W, B):
		Z = np.dot(W.T, X) + B
		A = self.sigmoid(Z)
		A = A > 0.5

		A = np.array(A, dtype= 'int64')
		Y = Y.T
		acc = (1 -np.sum(np.absolute(A - Y))/ Y.shape[1])*100
		rows, cols = Y.shape
		#print("y row is:", rows)
		#print("Y cols is:", cols)
		print("B is : ", B)
		acc = f"{acc:.2f}"
		print("Accuracy : ", acc, "%")

	def plot_cost_loss(self, iterations, cost_list):
		title_lines = [
		'Multi Class Logistic Regression',
		'Loss Descent',
		]
		title = '\n'.join(title_lines)

		plt.title(title, color='blue')
		plt.plot(np.arange(iterations), cost_list, color='red')
		plt.show()

def	main():
	path = "datasets/dataset_train.csv"
	iterations = 10000
	learning_rate = 0.0005
	lt = Logreg_train()
	lt.get_data_(path)
	lt.y_houses(path)
	X_train = lt.X_train()
	Y_train = lt.Y_train(X_train)
	W, B, cost_list = lt.models(X_train, Y_train, learning_rate=learning_rate, iterations=iterations)
	lt.save_weights(W, B)
	lt.accuracy(X_train, Y_train, W, B)
	lt.plot_cost_loss(iterations, cost_list)
	

if __name__ == "__main__":
    main()