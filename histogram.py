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
	filename: str
	db : Describe
	#numerical_data : list

	# Get data
	def import_data(self):
		self.dataset = self.db.open_data(self.path)
		return self.dataset

	#clean data
	def clean_data(self):
		self.db.distribution_NaN(self.dataset)
		self.db.del_NaN_column(self.dataset)
		cleaned_dataset = self.db.del_NaN_row(self.dataset)
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

#df_normalized.to_csv("datasets/df_normalized.csv", sep='\t', index=True)

#Creation of a class for statistics computation
@dataclass
class Statistiscal:
	db : Describe
	dn : Data_normalize
	unique_house_df: pd.DataFrame

	#calculate mean for each student
	def note_student_mean(self, df_normalized):
		db = Describe()
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
		df_mean.loc[:, 'Hogwarts House'] = cleaned_dataset.loc[df_mean.index, 'Hogwarts House']
		#df_mean.to_csv("datasets/df_mean.csv", sep='\t', index=True)
		return df_mean
	
	# Create unique house name
	def unique_house(self, df_mean):
		self.unique_house_df = pd.DataFrame({'Hogwarts House': df_mean['Hogwarts House'].unique()})
		unique_house_count = df_mean['Hogwarts House'].nunique()
		return unique_house_count
		
	# count number of student per house
	def count_student_per_house(self, df_mean):
		house_counts = df_mean['Hogwarts House'].value_counts().reset_index()
		house_counts.columns = ['Hogwarts House', 'count']
		self.unique_house_df = pd.merge(self.unique_house_df, house_counts, on='Hogwarts House', how='left')

		return self.unique_house_df
	
	# Sum all the notes of one house
	def sum_student_notes_per_house(self, df_mean):
		sum_notes = df_mean.groupby('Hogwarts House')['student mean'].sum().reset_index()
		sum_notes.columns = ['Hogwarts House', 'sum notes']
		self.unique_house_df = pd.merge(self.unique_house_df, sum_notes, on='Hogwarts House', how='left')
		return self.unique_house_df


	# Caluculate mean for each house
	def house_mean(self, df_mean):
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

#self.dn = Data_normalize()
@dataclass
class Histogram:
	data_normalizer: Data_normalize
	st : Statistiscal
	
	def hogwarts_histogram(self, unique_house_df):
		n_bins = unique_house_df.shape[0]  # Number of bins equal to the number of rows

		fig, ax = plt.subplots()  # Create figure and axis objects
		unique_house_df['std'] = unique_house_df['std'].astype(float) 

		house_colors = {
            'Ravenclaw': 'blue',
            'Slytherin': 'green',
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow'
        }

		for i, row in unique_house_df.iterrows():
			house = row['Hogwarts House']
			std_value = row['std']
			bin_range = (i, i + 1)  # Range for the bin positions on the x-axis

			# Plot histogram bars using plt.bar
			hist, bins = np.histogram(std_value, bins=n_bins)
			ax.bar(bin_range[0], std_value, width=1, alpha=0.7, label=house, color=house_colors[house])

		ax.set_xlabel(' ')
		ax.set_ylabel('Standard Deviation')
		ax.set_title('{}'.format(self.data_normalizer.filename))

		ax.set_xticks(range(unique_house_df.shape[0]))  # Set the x-ticks to correspond to the row index
		ax.set_xticklabels(unique_house_df['Hogwarts House'], rotation=45, ha='right')  # Set the x-tick labels to the house names

		ax.set_ylim(0, np.max(unique_house_df['std']))  # Set the y-axis limits based on the maximum value in std_values
	
		fig.tight_layout()
		plt.show()


def main():
	path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	unique_house_df = pd.DataFrame()  
	db = Describe()
	numerical_data = []
	dn = Data_normalize(path, filename, db)
	st = Statistiscal(db, dn, unique_house_df)
	htg = Histogram(dn, st)
	dataset = dn.import_data()
	cleaned_dataset = dn.clean_data()
	numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
	numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
	normalized_numerical_data = dn.normalizator(numerical_data)
	df_normalized = dn.index_to_normalized_data(normalized_numerical_data, numerical_columns, cleaned_dataset)
	df_mean = st.note_student_mean(df_normalized)
	df_mean = st.add_house_name(df_mean, cleaned_dataset)
	unique_house_count = st.unique_house(df_mean)
	unique_house_df = st.count_student_per_house(df_mean)
	sum_student_notes_per_house = st.sum_student_notes_per_house(df_mean)
	unique_house_df = st.house_mean(df_mean)
	unique_house_df = st.calculate_std(df_mean)
	with pd.option_context('display.max_rows', None):
		#print (unique_house_count.to_string(index=True))
		print (unique_house_df)
	htg.hogwarts_histogram(unique_house_df)

if __name__ == "__main__":
	main()
  
    
