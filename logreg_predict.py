import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pair_plot import Pairplot_graph
from describe import Describe
from histogram import Data_normalize
from histogram import Statistiscal
from logreg_train import Logreg_train

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
		W = W.T
		W = W.values
		x, y = W.shape
		print("W shape :", x, " X", y)
		X = X.T
		x, y = X.shape
		print("X shape :", x, " X", y)
		for col_idx in range(4):
			for row_idx in range(310):
				result = []
				i = 0
				
				# Working features per features
				for i in range(W.shape[0]):
					result = np.array
					result.append(W[i, col_idx] * X[row_idx, i])
					
				A = lt.sigmoid(result)
				max_value = max(A)
				A = [1 if val == max_value else 0 for val in A]
				A = np.array(result)
				print("A:", A)
				#x, y = A.shape
				print("A shape :", x, " X", y)
				Y[:, row_idx] += A

		np.set_printoptions(threshold=np.inf)
		print(Y.T)
		np.set_printoptions(threshold=310)


def main():
	path_W = "datasets/weights.csv"
	path_datatest = "datasets/dataset_test.csv"
	lp = Logreg_predict()
	lp.predict_(path_datatest, path_W)
	#lp.get_x(path_datatest)

if __name__ == "__main__":
	main()
