import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Describe:
	i : int = 0
	column : int = 0
	rows : int = 0
	dataset_only_digit: np.ndarray = None
	cleaned_dataset: np.ndarray = None

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
		rows = 0
		for _ in dataset.index:
			rows += 1
			self.rows = rows
		return self.rows
	
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
				dataset.drop(column, axis=1, inplace=True)
		return dataset

	#delete NaN: drop columns > 50% of NaN
	def del_NaN_column(self, dataset):
		rows = self.count_rows(dataset)
		NaN_distribution = pd.DataFrame(self.distribution_NaN(dataset))
		columns_to_drop = []
		for index, row in NaN_distribution.iterrows():
			if NaN_distribution[1][index] / rows > 0.5:
				columns_to_drop.append(NaN_distribution[0][index])
		dataset = dataset.drop(columns=columns_to_drop)
		return dataset

	#delete NaN: drop lines if NaN
	def del_NaN_row(self, dataset):
		cleaned_dataset = dataset.dropna(axis=0)
		return cleaned_dataset
	
	#init resume of cleaned_dataset
	def init_resume(self, cleaned_dataset):
		index_values = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
		resume = pd.DataFrame(columns=cleaned_dataset.columns, index=index_values)
		return resume

	def count_resume(self, resume):
		resume.loc['Count',:] = self.rows

	def mean_resume(self, cleaned_dataset, resume):
		for column in cleaned_dataset.columns:
			column_data = cleaned_dataset[column]
			column_sum = sum(column_data)
			mean = column_sum / self.rows
			resume.loc['Mean', column] = mean
		print("mean = \n", mean)
	
	def std_resume(self, cleaned_dataset, resume):
		for column in cleaned_dataset.columns:
			mean = resume.loc['Mean', column]
			deviation_sum = 0
			for index, row in cleaned_dataset.iterrows():
				deviation_sum += (row[column] - mean) ** 2
				variance = deviation_sum / self.rows
				std = variance ** 0.5
				resume.loc['Std', column] = std

	def min_resume(self, cleaned_dataset, resume):
		for column in cleaned_dataset.columns:
			min = cleaned_dataset.iloc[1][column]
			for index, row in cleaned_dataset.iterrows():
				if row[column] < min:
					min = row[column]
			resume.loc['Min', column] = min

	def max_resume(self, cleaned_dataset, resume):
		for column in cleaned_dataset.columns:
			min = cleaned_dataset.iloc[1][column]
			for index, row in cleaned_dataset.iterrows():
				if row[column] > min:
					min = row[column]
			resume.loc['Max', column] = min

	'''def quartiles_resume(self, cleaned_dataset, resume):
		for column in cleaned_dataset.columns:
			values = sorted(cleaned_dataset[column])
			n = self.rows
			q25_index = int(n // 4)
			q50_index = int(n // 2)
			q75_index = int(3 * n // 4)
			q25 = values[q25_index]
			q50 = values[q50_index]
			q75 = values[q75_index]
			resume.loc['25%', column] = q25
			resume.loc['50%', column] = q50
			resume.loc['75%', column] = q75'''


	
def main():

	db = Describe(dataset_only_digit=None)
	dataset = db.open_data("datasets/dataset_test.csv")
	i = db.count_column(dataset)
	j = db.count_rows(dataset)
	l = db.distribution_NaN(dataset)
	m = db.del_NaN_column(dataset)
	n = db.del_NaN_row(m)
	#k = db.check_digits(n)
	#o = db.init_resume(k)
	#p = db.count_resume(o)
	#mean = db.mean_resume(k, o)
	#std = db.std_resume(k, o)
	#min = db.min_resume(k, o)
	#max = db.max_resume(k, o)
	#quart = db.quartiles_resume(k, o)
	#print("Is NaN distribution\n", l)
	#print("Number of column:", i)
	#print("Number of rows:", j)
	#print("cleaned_dataset:\n", k)
	#print("del_NaN_column\n", m)
	#print("resume mean:\n", o)
	#print("resume std:\n", o)
	

if __name__ == "__main__":
    main()
        