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
	
	#delete NaN: drop columns > 50% of NaN
	def del_NaN_column(self, dataset):
		rows = dataset.shape[0]
		NaN_distribution = pd.DataFrame(self.distribution_NaN(dataset))
		columns_to_drop = []
		for index, row in NaN_distribution.iterrows():
			if NaN_distribution[1][index] / rows > 0.5:
				columns_to_drop.append(NaN_distribution[0][index])
				dataset = dataset.drop(columns=columns_to_drop, axis=1)
		return dataset

	#delete NaN: drop lines if NaN
	def del_NaN_row(self, dataset):
		dataset = pd.DataFrame(dataset)
		cleaned_dataset = dataset.dropna(axis=0)
		return cleaned_dataset
	
	#select columns filled with digits
	def check_digits(self, cleaned_dataset):
		digit_columns = []
		for column in cleaned_dataset.columns:
			is_only_digits = cleaned_dataset[column].apply(lambda x: isinstance(x, (float, int)))
			if is_only_digits.all() and cleaned_dataset[column].notnull().all():
				digit_columns.append(column)
			is_only_digits = cleaned_dataset[digit_columns]
		return is_only_digits
	
	#init resume of is_only_digits dataset
	def init_resume(self, is_only_digits):
		index_values = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
		resume = pd.DataFrame(columns=is_only_digits.columns, index=index_values)
		return resume

	#Put number of rows in resume tab
	def count_resume(self, resume):
		resume.loc['Count',:] = self.rows

	#Mean
	def mean_resume(self, is_only_digits, resume):
		for column in is_only_digits.columns:
			column_data = is_only_digits[column]
			column_sum = sum(column_data)
			mean = column_sum / self.rows
			resume.loc['Mean', column] = mean
			
	#(sum of the square of the deviations from the mean)**0.5
	def std_resume(self, is_only_digits, resume):
		for column in is_only_digits.columns:
			mean = resume.loc['Mean', column]
			deviation_sum = 0
			for index, row in is_only_digits.iterrows():
				deviation_sum += (row[column] - mean) ** 2
				variance = deviation_sum / self.rows
				std = variance ** 0.5
				resume.loc['Std', column] = std

	#Min value in resume tab
	def min_resume(self, is_only_digits, resume):
		for column in is_only_digits.columns:
			min = is_only_digits.iloc[1][column]
			for index, row in is_only_digits.iterrows():
				if row[column] < min:
					min = row[column]
			resume.loc['Min', column] = min

	#Max value in resume tab
	def max_resume(self, is_only_digits, resume):
		for column in is_only_digits.columns:
			max = is_only_digits.iloc[1][column]
			for index, row in is_only_digits.iterrows():
				if row[column] > max:
					max = row[column]
			resume.loc['Max', column] = max

	#3 quartiels 25, 50 75%
	def quartiles_resume(self, is_only_digits, resume):
		for column in is_only_digits.columns:
			values = sorted(is_only_digits[column])
			n = self.rows
			q25_index = int(n // 4)
			q50_index = int(n // 2)
			q75_index = int(3 * n // 4)
			q25 = values[q25_index]
			q50 = values[q50_index]
			q75 = values[q75_index]
			resume.loc['25%', column] = q25
			resume.loc['50%', column] = q50
			resume.loc['75%', column] = q75
	
	#generate tab resume (easier than to put every thing in the main)
	def generate_resume(self, dataset):
		i = self.count_column(dataset)
		j = self.count_rows(dataset)
		l = self.distribution_NaN(dataset)
		m = self.del_NaN_column(dataset)
		cleaned_dataset = self.del_NaN_row(m)
		is_only_digits = self.check_digits(cleaned_dataset)
		resume = self.init_resume(is_only_digits)
		p = self.count_resume(resume)
		mean = self.mean_resume(is_only_digits, resume)
		std = self.std_resume(is_only_digits, resume)
		min = self.min_resume(is_only_digits, resume)
		max = self.max_resume(is_only_digits, resume)
		quart = self.quartiles_resume(is_only_digits, resume)
		return resume
	
def main():

	db = Describe(dataset_only_digit=None)
	dataset = db.open_data("datasets/dataset_train.csv")
	resume = db.generate_resume(dataset)
	print("resume :\n", resume)

if __name__ == "__main__":
    main()
        