import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pair_plot import Pairplot_graph
from describe import Describe
from histogram import Data_normalize
from histogram import Statistiscal
from logreg_train import Logreg_train
import csv

@dataclass
class Logreg_predict:
	datatest : np.ndarray = None
	W : np.ndarray = None

	def get_x(self, path):
		db = Describe()
		dn = Data_normalize(path, db)
		st = Statistiscal(db, dn)
		dataset = dn.import_data()
		cleaned_dataset = dn.clean_data(dataset)
		numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
		numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
		X = dn.normalizator(numerical_data)
		print(X)
		x, y = X.shape
		print("X shape :", x, " X", y)
		return X

	def predict_(self, path_datatest, path_W):
		lt = Logreg_train()
		X = self.get_x(path_datatest)
		print("dataset : ", X)
		W = pd.read_csv(path_W)
		print("W : ", W)

		#Y = np.dot(W.T, X.T) -1.1052449834420803
		Y = np.zeros((4, 310))
		W = W.values
		x, y = W.shape
		print("W shape :", x, " X", y)
		x, y = X.shape
		print("X shape :", x, " X", y)
		Y = np.dot(X, W) -1.105582243394336
		print("Y is : ", Y)
		x, y = Y.shape
		print("Y shape :", x, " X", y)
		A = lt.sigmoid(Y)
		x, y = A.shape
		print("A shape :", x, " X", y)
		result = []
		for row in A:
			max_value = max(row)
			new_row = [1 if val == max_value else 0 for val in row]
			result.append(new_row)
				
		np.set_printoptions(threshold=np.inf)
		result = np.array(result)
		#print("result: ", result)
		np.set_printoptions(threshold=310)
		return result

	def convert_to_house_names(self, path_datatest, path_W):
		result = self.predict_(path_datatest, path_W)
		house_list = np.array(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']).reshape(-1, 1)
		x, y = result.shape
		print("result shape :", x, " X", y)
		x, y = house_list.shape
		print("house_list shape :", x, " X", y)
		prediction = []
		for i in range(result.shape[0]):
			houses = [i]
			for j in range(result.shape[1]):
				if result[i, j] == 1:
					houses.append(house_list[j, 0])
			prediction.append(houses)

		prediction = np.array(prediction)	
		print(prediction[:10])
		return prediction
	
	def save_csv(self, save_path, data_list):
		with open(save_path, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(['Index', ' Hogwart Houses'])
			csvwriter.writerows(data_list)

def main():
	path_W = "datasets/weights.csv"
	path_datatest = "datasets/dataset_test.csv"
	save_path = "datasets/houses.csv"
	lp = Logreg_predict()
	#lp.predict_(path_datatest, path_W)
	#lp.get_x(path_datatest)
	data_list = lp.convert_to_house_names(path_datatest, path_W)
	lp.save_csv(save_path, data_list)

if __name__ == "__main__":
	main()
