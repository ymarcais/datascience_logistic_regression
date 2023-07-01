import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass
from describe import Describe


@dataclass
class Histogram:
	

	def import_data(self):
		db = Describe()
		dataset = db.open_data("datasets/dataset_test.csv")
		df = db.generate_resume(dataset) 
		return df

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
		ax.set_title('Histogram of "std" values')
		#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(df.columns))
		ax.set_xticks(range(len(df.columns)))  # Set the x-ticks to correspond to the column index
		ax.set_xticklabels(df.columns, rotation=45, ha='right')  # Set the x-tick labels to the column names
		ax.set_ylim(0, np.max(std_values))  # Set the y-axis limits based on the maximum value in std_values
		ax.set_yscale('log', base=1.01) 

		

		
		fig.tight_layout()
		plt.show()

def main():
	htg = Histogram()
	htg.hogwarts_histogram()

if __name__ == "__main__":
    main()


    
    
