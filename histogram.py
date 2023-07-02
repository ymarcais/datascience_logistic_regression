import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass
from describe import Describe

@dataclass
class Data_normalize():
	path	: str
	filename: str

	# Get data
	def import_data(self):
		db = Describe()
		self.dataset = db.open_data(self.path)
		self.file_name = path.split("/")[-1]
			
	# Separate copy of the numerical columns
	def separate_numerical(self, dataset)
		numerical_columns = self.dataset.select_dtypes(includ=['int', 'float']).columns
		self.numerical_data = self.dataset[numerical_columns].copy()

	# StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1
	# fit() calculate the mean and the std, transform() applies to the data
	def normalizator(self)
		scaler = StandardScaler()
		scaler.fit(self.numerical_data)
		self.normalized_data = sacler.transform(self.numerical_data)
	
	# Create a new DataFrame with the normalized numerical data and original index from dataset
	def index_to_normalized_data(self)
		df_normalized = pd.DataFrame(self.normalized_numerical_data, columns=self.numerical_columns, index=self.dataset.index)
	
#todoclass Stastistical():

	# From normalized notes, calculate mean per student	

	#need to use groupby to calculate std per house
	team_std = df_values.groupby(df_teams['house']).apply(lambda x: x.std())



dn = Data_normalize()
@dataclass
class Histogram:

	def hogwarts_histogram(self):
		df = self.import_data()
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
	htg = Histogram(path)
	htg.hogwarts_histogram()

if __name__ == "__main__":
    main()


    
    
