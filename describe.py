import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Describe:
	i : int = 0
	column : int = 0
	rows : int = 0
	dataset_only_digit: np.ndarray = None

	#get data
	def open_data(self, path):
		dataset = pd.read_csv(path)
		return dataset.set_index("Index")

	#count columns
	def count_column(self, dataset):
		count = 0
		for _ in dataset.columns:
			count += 1
		return count
	
	#count rows
	def count_rows(self, dataset):
		count = 0
		for _ in dataset:
			count += 1
		return count
	
	#count NaN distribution by column
	def distribution_NaN(self, dataset):
		NaN_distribution = []
		for idx, column in enumerate(dataset.columns):
			num_NaN = dataset[column].apply(lambda x: pd.isna(x)).sum()
			if num_NaN > 0:
				NaN_distribution.append([column, num_NaN])
		return NaN_distribution
	
	#check if columns are filled with digits
	def check_digits(self, dataset):
		digit_columns = {}
		for column in dataset.columns:
			is_only_digits = dataset[column].apply(lambda x: type(x) == float or type(x) == int)
			if is_only_digits.all() and dataset[column].notnull().all():
				digit_columns[column] = True
			else:
				digit_columns[column] = False
		return digit_columns

	#delete NaN: drop columns > 50% of NaN
	def del_NaN_column(self, dataset):
		rows = self.count_rows(dataset)
		NaN_distribution = pd.DataFrame(self.distribution_NaN(dataset))
		columns_to_drop = []
		for index, row in NaN_distribution.iterrows():
			if NaN_distribution[1][index] / rows > 0.5:
				columns_to_drop.append(NaN_distribution[0][index])
		dataset.drop(columns=columns_to_drop, inplace=True)
		return dataset

	#delete NaN: drop lines if NaN
	'''def del_Nan_row(self, dataset, int i = 0)
		for row in dataset.rows:
			num_NaN = dataset[column].apply(lambda x: pd.isna(x)).sum()
			if num_NaN :
				dataset.drop(i)
		return dataset'''

def main():

	db = Describe(dataset_only_digit=None)
	dataset = db.open_data("datasets/dataset_test.csv")
	i = db.count_column(dataset)
	j = db.count_rows(dataset)
	k = db.check_digits(dataset)
	l = db.distribution_NaN(dataset)
	m = db.del_NaN_column(dataset)
	#print("Is NaN distribution\n", l)
	#print("Number of column0:", i)
	#print("Number of Rows:", j)
	#print("Dataset with Only Digits:", k)
	print("del_NaN_column\n", m)

if __name__ == "__main__":
    main()
        