import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Describe:
	i : int = 0
	column : int = 0
	rows : int = 0
	dataset_only_digit: np.ndarray = None

	def open_data(self, path):
		dataset = pd.read_csv(path)
		return dataset

	def check_features(self, dataset):
		for _ in dataset.columns:
			self.column += 1
		return self.column
	
	def check_rows(self, dataset):
		for _ in dataset.iterrows():
			self.rows += 1
		return self.rows
	
	def check_digits(self, dataset):
		is_only_digits = dataset.applymap(lambda x: str(x).isdigit())
		self.dataset_only_digit = dataset[is_only_digits.all(axis=1)].values
		return self.dataset_only_digit
	
def main():

	db = Describe(dataset_only_digit=None)
	dataset = db.open_data("C:/Users/ymarc/Documents/Python/demenagement/datasets/dataset_test.csv")
	i = db.check_features(dataset)
	j = db.check_rows(dataset)
	k = db.check_digits(dataset)
	print("Number of Features:", i)
	print("Number of Rows:", j)
	print("Dataset with Only Digits:", k)

if __name__ == "__main__":
    main()
        