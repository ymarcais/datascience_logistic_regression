import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass
from describe import Describe

@dataclass
class Data_normalize:
	path	: str
	filename: str
	db : Describe()

	# Get data
	def import_data(self):
		self.dataset = db.open_data(self.path)
		return self.dataset

	#clean data
	def clean_data(self)
		db.distribution_NaN(dataset)
		dataset = db.del_NaN_column(dataset)
		cleaned_dataset = db.del_NaN_row(dataset)
		return cleaned_dataset
			
	# Separate copy of the numerical columns
	def separate_numerical(self, cleaned_dataset):
		numerical_columns = self.cleaned_dataset.select_dtypes(includ=['int', 'float']).columns
		self.numerical_data = self.cleaned_dataset[numerical_columns].copy()

	# StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1
	# fit() calculate the mean and the std, transform() applies to the data
	def normalizator(self):
		scaler = StandardScaler()
		scaler.fit(self.numerical_data)
		self.normalized_data = scaler.transform(self.numerical_data)
	
	# Create a new DataFrame with the normalized numerical data and original index from cleaned_dataset
	def index_to_normalized_data(self):
		df_normalized = pd.DataFrame(self.normalized_numerical_data, columns=self.numerical_columns, index=self.cleaned_dataset.index)
		return df_normalized

#todoclass Stastistical(): -la
@dataclass
class Statistiscal:
	db : Describe()
	dn : Data_normalize()

	#calculate mean for each student
	def note_student_mean(sef):
		db = describe()
		count = self.deb.count(dn.df_normalized)
		df_mean = np.DataFrame()
		for index, row in dn.df_normalized.iterrows():
			total = 0
			for col in dn.df_normalized.columns:
				total = row[col]
			mean = total / count
			df_mean['Index'] = df_normalized.index
			df_mean[index,'student mean'] = mean

	# add house name of house to df_mean dataframe (using index as reference)
	def add_house_name(self):
		df_mean['Hogwarts house'] = cleaned_dataset.loc[df1.index, 'Hogwarts house']
		return df_mean
	
	# Create unique house name
	def unique_house(self, df_mean)
		unique_house_df = pd.DataFrame({'Hogwarts house': df_mean['Hogwarts house'].unique()})

	# count number of student per house
	def count_student_per_house(self, df_mean)
		house_counts = df_mean['Hogwarts house'].value_counts().reset_index()
		house_counts.columns = ['Hogwarts house', 'count']
		unique_house_df = pd.merge(unique_house_df, house_counts, on='names', how='left')
		return unique_house_df
	
	# Sum all the notes of one house
	def sum_student_notes_per_house(self, def_mean, unique_house_df)
		sum_notes = df_mean['Hogwarts house'].sum().reset_index()
		sum_notes.columns = ['Hogwarts house', 'sum notes']
		unique_house_df = pd.merge(unique_house_df, sum_notes, on='names', how='left')
		return unique_house_df

	# Caluculate mean for each house
	def note_mean_per_house(self,def_mean, unique_house_df)
		unique_house_df['house mean'] = unique_house_df['sum notes'] / unique_house_df['count']
		return unique_house_df

	# Caulculate std for each house
	
	

	
	

 	&& calulate std per house
			house_std = df_values.groupby(df_teams['Hogwarts house']).apply(lambda x: x.std())

	




	# From normalized notes, calculate mean per student	

	#need to use groupby to calculate std per house
	#team_std = df_values.groupby(df_teams['house']).apply(lambda x: x.std())



#dn = Data_normalize()
@dataclass
class Histogram:
	data_normalizer: Data_normalize

	def hogwarts_histogram(self, path):
		df = self.data_normalizer.import_data()
		std_values = df.loc['Std'].values
		n_bins = len(df.columns)  # Number of bins equal to the number of columns
		
		fig, ax = plt.subplots()  # Create figure and axis objects
		
		for i, column in enumerate(df.columns):
			column_values = df[column].values
			bin_range = (i, i+1)  # Range for the bin positions on the x-axis
			
			# Plot histogram bars using plt.bar
			hist, bins = np.histogram(column_values, bins=n_bins)
			ax.bar(bin_range[0], std_values[i], width=1, alpha=0.7, label=column)
		
		ax.set_xlabel(' ')
		ax.set_ylabel('Value')
		ax.set_title('{}'.format(self.dn.filename))
		#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(df.columns))
		ax.set_xticks(range(len(df.columns)))  # Set the x-ticks to correspond to the column index
		ax.set_xticklabels(df.columns, rotation=45, ha='right')  # Set the x-tick labels to the column names
		ax.set_ylim(0, np.max(std_values))  # Set the y-axis limits based on the maximum value in std_values
		ax.set_yscale('log', base=1.01) 

		fig.tight_layout()
		plt.show()

def main():
	path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	dn = Data_normalize(path, filename)
	htg = Histogram(dn)
	htg.hogwarts_histogram(path)

if __name__ == "__main__":
    main()


    
    
