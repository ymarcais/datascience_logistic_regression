import numpy as np
import pandas as pd
from histogram import Data_normalize 
from describe import Describe
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

@dataclass
class R_correlation:
	db : Describe
	dn : Data_normalize
	r_array : list
	df_normalized : list
	count_rows : int = 0
	last_r : int = 0

	# import data, clean and normalized using previous describe and histogram's classes
	def get_normalized_data(self):
		numerical_data = []
		dataset = self.dn.import_data()
		cleaned_dataset = self.dn.clean_data()
		numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
		numerical_data = self.dn.separate_numerical(cleaned_dataset, numerical_columns)
		#normalized_numerical_data = self.dn.normalizator(numerical_data)
		normalized_numerical_data = numerical_data # To delete after test
		self.df_normalized = self.dn.index_to_normalized_data(normalized_numerical_data, numerical_columns, cleaned_dataset)
		#self.df_normalized.to_csv("df_normalized.csv", sep='\t', index=True)
		return self.df_normalized
	
	# One way to found similarities is to use Pearson's correlation which seems the most relevent related to the subject.
	# Pearson's correlation : r = (Σ((X - X̄) * (Y - Ȳ))) / sqrt(Σ((X - X̄)^2) * Σ((Y - Ȳ)^2))

	#Double loops for iterate X and Y columns

	#mean of the feature
	def mean_column(self, X, count_rows):
		#self.count_rows = self.db.count_rows(self.df_normalized)
		total = 0
		x_mean = 0
		if X < self.df_normalized.shape[1]:
			for _, row in self.df_normalized.iterrows():
				total += row.iloc[X]
		x_mean = total / count_rows
		return x_mean
	
	# apply r correlation formulat r = (Σ((X - X̄) * (Y - Ȳ))) / sqrt(Σ((X - X̄)^2) * Σ((Y - Ȳ)^2))
	def r_correlation(self, X, x_mean, Y, y_mean):
		r1 = 0
		r2 = 0
		r = 0
		for _, row in self.df_normalized.iterrows():
			if X < self.df_normalized.shape[1] - 1 and Y < self.df_normalized.shape[1] :
				r1 += ((row.iloc[X] - x_mean) * (row.iloc[Y] - y_mean))
				r2 += ((row.iloc[X] - x_mean) ** 2) * np.sum((row.iloc[Y] - y_mean) ** 2)
		r2 = r2 ** 0.5
		if r2 == 0:
			return None
		r = r1 / r2
		return float(f'{r:.4f}')
	
	#check and store // r = 0.9 and r = 1.1 are equidistent to 1
	def	check_best_r(self, new_r, X, Y):
		if (abs(self.last_r - 1) > abs(new_r - 1)):
			self.last_r = new_r
			self.r_array = []
			self.r_array.append((self.last_r, X, Y))
		elif (abs(self.last_r - 1) == abs(new_r - 1)):
			self.r_array.append((self.last_r, X, Y))
		return self.r_array 

	# iterrate all columns pairs with correlation of AB = BA
	def r_itteration_in_df_normalized(self, count_rows):
		count_columns  = 0
		i = 0
		count_columns = int(self.db.count_column(self.df_normalized))
		for X, column in enumerate(self.df_normalized.columns):
			if X == count_columns:
				break
			for Y, next_column in enumerate(self.df_normalized.columns[X + 1:], start=X + 1):
				x_mean = self.mean_column(X, count_rows)
				y_mean = self.mean_column(Y, count_rows)
				new_r = self.r_correlation(X, x_mean, Y, y_mean)
				self.r_array = self.check_best_r(new_r, X, Y)
				i += 1
				print(f" r{i} : {new_r}")
				#self.r_array = np.array(self.r_array).reshape(-1, 3)
		return self.r_array
	
@dataclass
class Plot_2_correlated_columns:

	def scatter_columns(self, df_normalized, r_array):
				
		correlation_coefficient = r_array[0][0]
		column_index1 = r_array[0][1]
		column_index2 = r_array[0][2]
		
		# Retrieve the column names from the DataFrame using iloc
		column_names = [df_normalized.columns[num] for num in [column_index1, column_index2]]

		# Get the data for the two selected columns
		x_data = df_normalized.index
		y_data1 = df_normalized.iloc[:, column_index1] 
		y_data2 = df_normalized.iloc[:, column_index2]

		# Use two colors for the two columns
		colors = ['red', 'blue']
		plt.scatter(x_data, y_data1, c=colors[0], label=column_names[0])
		plt.scatter(x_data, y_data2, c=colors[1], label=column_names[1])
		plt.xlabel("Students")
		plt.ylabel("Notes")
		plt.title('What are the two features that are similar ?')
		plt.legend(loc='upper left', bbox_to_anchor=(0, -0.05), borderaxespad=0.)
		plt.text(1, -0.1, f'r = {correlation_coefficient}, Features: {column_names[0]} & {column_names[1]}',
				horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
		plt.show()

def main():
	path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	db = Describe()
	dn = Data_normalize(path, filename, db)
	rc = R_correlation(db, dn, r_array=[], df_normalized=[], count_rows = 0, last_r = 0)
	df_normalized = rc.get_normalized_data()
	count_rows = db.count_rows(df_normalized)
	r_array = rc.r_itteration_in_df_normalized(count_rows)
	print(r_array)
	pt = Plot_2_correlated_columns()
	pt.scatter_columns(df_normalized, r_array)

if __name__ == "__main__":
	main()

