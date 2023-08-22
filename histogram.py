import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass
from describe import Describe
from sklearn.preprocessing import StandardScaler

@dataclass
class Data_normalize:
	path	: str
	#filename: str
	db : Describe
	#numerical_data : list

	# Get data
	def import_data(self):
		self.dataset = self.db.open_data(self.path)
		return self.dataset

	#clean data
	def clean_data(self, dataset):
		cleaned_dataset = self.db.del_NaN_column(dataset)
		cleaned_dataset = self.db.del_NaN_row(cleaned_dataset)
		return cleaned_dataset
			
	# Separate copy of the numerical columns
	def separate_numerical(self, cleaned_dataset, numerical_columns):
		numerical_data = cleaned_dataset[numerical_columns]
		return numerical_data

	# StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1
	# fit() calculate the mean and the std, transform() applies to the data
	def normalizator(self, numerical_data):
		scaler = StandardScaler()
		scaler.fit(numerical_data)
		normalized_numerical_data = scaler.transform(numerical_data)
		return normalized_numerical_data
	
	# Add original index to Create a new DataFrame with the normalized numerical data and original index from cleaned_dataset
	def index_to_normalized_data(self, normalized_numerical_data, numerical_columns, cleaned_dataset):
		df_normalized = pd.DataFrame(normalized_numerical_data, columns=numerical_columns, index=cleaned_dataset.index)
		return df_normalized
	
	# Mother function
	def data_normalize_(self):
		self.dataset = self.import_data()
		self.cleaned_dataset = self.clean_data(self.dataset)
		#print("self cleanded dataset", self.cleaned_dataset)
		numerical_columns = self.cleaned_dataset.select_dtypes(include=['int', 'float']).columns
		numerical_data = self.separate_numerical(self.cleaned_dataset, numerical_columns)
		normalized_numerical_data = self.normalizator(numerical_data)
		df_normalized = self.index_to_normalized_data(normalized_numerical_data, numerical_columns, self.cleaned_dataset)
		return df_normalized
	
#df_normalized.to_csv("datasets/df_normalized.csv", sep='\t', index=True)

#Creation of a class for statistics computation
@dataclass
class Statistiscal:
	db : Describe
	dn : Data_normalize
	#unique_house_df: pd.DataFrame

	#calculate mean for each student
	def note_student_mean(self, df_normalized):
		count = self.db.count_column(df_normalized)
		df_mean = pd.DataFrame()
		df_mean['Index'] = df_normalized.index
		df_mean = df_mean.set_index('Index', drop = True)
		for index, row in df_normalized.iterrows():
			total = 0
			for col in df_normalized.columns:
				total += row[col]
			mean = total / count
			df_mean.loc[index, 'student mean'] = mean
		return df_mean

	# add house name of house to df_mean dataframe (using index as reference)
	def add_house_name(self, df_mean, cleaned_dataset):
		df_mean_house = pd.DataFrame()
		df_mean_house = df_mean
		df_mean_house.loc[:, 'Hogwarts House'] = cleaned_dataset.loc[df_mean.index, 'Hogwarts House']
		column_to_move = df_mean_house['Hogwarts House']
		df_mean_house.drop(columns='Hogwarts House', inplace=True)
		df_mean_house.insert(0, 'Hogwarts House', column_to_move)
		print(df_mean_house[:10])
		return df_mean_house
	
	# Create unique house name
	def unique_house(self, df_mean_house):
		self.unique_house_df = pd.DataFrame({'Hogwarts House': df_mean_house['Hogwarts House'].unique()})
		
	# count number of student per house
	def count_student_per_house(self, df_mean_house):
		house_counts = df_mean_house['Hogwarts House'].value_counts().reset_index()
		house_counts.columns = ['Hogwarts House', 'count']
		self.unique_house_df = pd.merge(self.unique_house_df, house_counts, on='Hogwarts House', how='left')
		return self.unique_house_df
	
	# Sum all the notes of one house
	def sum_student_notes_per_house(self, df_mean_house):
		sum_notes = df_mean_house.groupby('Hogwarts House')['student mean'].sum().reset_index()
		sum_notes.columns = ['Hogwarts House', 'sum notes']
		self.unique_house_df = pd.merge(self.unique_house_df, sum_notes, on='Hogwarts House', how='left')
		return self.unique_house_df

	# Caluculate mean for each house
	def house_mean(self):
		self.unique_house_df['house mean'] = self.unique_house_df['sum notes'] / self.unique_house_df['count']
		return self.unique_house_df

	# Calculate std whithout std()
	def calculate_std(self, df_mean):
		for house in self.unique_house_df['Hogwarts House']:
			students_mean = df_mean[df_mean['Hogwarts House'] == house]['student mean']
			house_counts = self.unique_house_df[self.unique_house_df['Hogwarts House'] == house]['count']
			house_mean = self.unique_house_df[self.unique_house_df['Hogwarts House'] == house]['house mean']
			std = 0
			for student_mean in students_mean:
				std += (student_mean - house_mean)**2
			std = (std / (house_counts - 1))** 0.5
			self.unique_house_df.loc[self.unique_house_df['Hogwarts House'] == house, 'std'] = std
		return self.unique_house_df
	
	# Mother function
	def statistical_(self, df_normalized):
		df_mean = self.note_student_mean(df_normalized)
		df_mean_house = self.add_house_name(df_mean, self.dn.cleaned_dataset)
		self.unique_house(df_mean_house)
		self.unique_house_df = self.count_student_per_house(df_mean_house)
		self.sum_student_notes_per_house(df_mean_house)
		self.unique_house_df = self.house_mean()
		self.unique_house_df = self.calculate_std(df_mean)
		with pd.option_context('display.max_rows', None):
			print (self.unique_house_df)
		

#self.dn = Data_normalize()
@dataclass
class Histogram:
	st : Statistiscal
	
	def hogwarts_histogram(self, path):
		filename = path.split("/")[-1]
		n_bins = self.st.unique_house_df.shape[0]  # Number of bins equal to the number of rows

		fig, ax = plt.subplots()  # Create figure and axis objects
		self.st.unique_house_df['std'] = self.st.unique_house_df['std'].astype(float) 

		house_colors = {
            'Ravenclaw': 'blue',
            'Slytherin': 'green',
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow'
        }

		for i, row in self.st.unique_house_df.iterrows():
			house = row['Hogwarts House']
			std_value = row['std']
			bin_range = (i, i + 1)  # Range for the bin positions on the x-axis

			# Plot histogram bars using plt.bar
			hist, bins = np.histogram(std_value, bins=n_bins)
			ax.bar(bin_range[0], std_value, width=1, alpha=0.7, label=house, color=house_colors[house])

		ax.set_xlabel(' ')
		ax.set_ylabel('Standard Deviation')
		ax.set_title('{}'.format(filename))

		ax.set_xticks(range(self.st.unique_house_df.shape[0]))  # Set the x-ticks to correspond to the row index
		ax.set_xticklabels(self.st.unique_house_df['Hogwarts House'], rotation=45, ha='right')  # Set the x-tick labels to the house names

		ax.set_ylim(0, np.max(self.st.unique_house_df['std']))  # Set the y-axis limits based on the maximum value in std_values
	
		fig.tight_layout()
		plt.show()


def main():
	path = "datasets/dataset_train.csv"
	db = Describe()
	dn = Data_normalize(path, db)
	st = Statistiscal(db, dn)
	htg = Histogram(st)
	df_normalized = dn.data_normalize_()
	st.statistical_(df_normalized)
	htg.hogwarts_histogram(path)

if __name__ == "__main__":
	main()
  
    
