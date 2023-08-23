import pandas as pd
import numpy as np
from dataclasses import dataclass
from describe import Describe
from histogram import Data_normalize
from histogram import Statistiscal
from logreg_train import Logreg_train
import csv

@dataclass
class Logreg_predict:
	datatest : np.ndarray = None
	W : np.ndarray = None

	#get file to predict
	def get_x(self, path):
		db = Describe()
		dn = Data_normalize(path, db)
		st = Statistiscal(db, dn)
		dataset = dn.import_data()
		dataset = pd.DataFrame(dataset)
		
		cleaned_dataset = dn.clean_data(dataset)
		
		numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
		numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
		X = dn.normalizator(numerical_data)
		return X
	
	#extract W and B from CSV file
	def get_W_B(self, path_W):
		data = pd.read_csv(path_W, header=None)
		W = data.iloc[1:, :4]
		W = W.astype(np.float64)
		B = data.iloc[1, 4]
		B = float(B)
		return W, B

	#get prediction matrix filled with one 1 per line
	def predict_(self, path_datatest, path_W):
		lt = Logreg_train()
		X = self.get_x(path_datatest)

		W, B = self.get_W_B(path_W)

		Y = np.zeros((4, 310))
		
		Y = np.dot(X, W) + B
		A = lt.sigmoid(Y)
		
		result = []
		for row in A:
			max_value = max(row)
			new_row = [1 if val == max_value else 0 for val in row]
			result.append(new_row)
				
		np.set_printoptions(threshold=np.inf)
		result = np.array(result)
		np.set_printoptions(threshold=310)
		return result

	#convert results 1 to house name
	def convert_to_house_names(self, path_datatest, path_W):
		result = self.predict_(path_datatest, path_W)
		house_list = np.array(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']).reshape(-1, 1)
		
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
	
	#save index and house name to csv
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
	data_list = lp.convert_to_house_names(path_datatest, path_W)
	lp.save_csv(save_path, data_list)

if __name__ == "__main__":
	main()

