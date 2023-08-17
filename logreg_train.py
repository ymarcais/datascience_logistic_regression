import pandas as pd
import numpy as np
from dataclasses import dataclass, field
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/mnt/nfs/homes/ymarcais/ft_linear_regression')
#from gradient_descent import GradientDescent
from describe import Describe
from histogram import Data_normalize
from histogram import Statistiscal
from scatter_plot	import R_correlation
from pair_plot import Pairplot_graph

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
			#(df['Hogwarts House'] == house) creates a boolean Series
			#astype(int)  converts True to 1 and False to 0.
			self.dataframe[column_name] = (self.dataframe['Hogwarts House'] == house).astype(int)
	
	#get X_train and reshape
	def X_train(self):
		X_train = self.dataframe.iloc[:, 1:14].values
		X_train = X_train.T
		print(X_train)
		return X_train
	
	#get Y_train and reshape
	def Y_train(self, X_train):
		Y_train = self.dataframe.iloc[:, 14:18].values
		Y_train = Y_train.reshape(4, X_train.shape[1])
		print(Y_train)
		#return Y_train

	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def models(self, X, Y, learning_rate, iterations)
		m = X.shape[1]
		n = X_train.shape[0]

		W = np.zeros((n, 1))
		B = 0

		for i in range(iterations):
			Z = np.dot(W.T, X_train) + B
			A = self.sigmoid(Z)
		
			cost = -(1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
			dW = (1/m) * np.dot(A - Y, X.T)
			dB =  (1/m) * np.sum(A - Y)

			W -= learning_rate * dW.T
			B -= learning_rate * dB

			cost_list.append(cost)

			if(i % (iterations / 10 ) == 0):
				print("cost after :", i, "iteration is : ", cost)

		return W, B, cost_list

	


def	main():
	path = "datasets/dataset_train.csv"
	iterations = 10000
	learning_rate = 0.0005
	lt = Logreg_train()
	lt.get_data_(path)
	lt.y_houses(path)
	X_train = lt.X_train()
	lt.Y_train(X_train)
	W, Y, cost_list = models(X_train, Y_train,  learning_rate = learning_rate, iterations = iterations)

if __name__ == "__main__":
    main()