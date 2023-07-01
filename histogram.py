import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass

@dataclass
class Histogram:
    
	def import_data(self):
		dataset = open_data("datasets/dataset_test.csv")
		df = generate_resume(dataset) 
		return df

	def hogwarts_histogram(self):
		std_values = df.loc['Std'].values
		n_bins = 'auto'
		column_values = df[column].values
		_, bins, _ = ax.hist(column_values, bins=n_bins, alpha=0.7, label=column)
		ax.set_xlabel('Value')
		ax.set_ylabel('Frequency')
		ax.set_title('Histogram of "std" values')
		ax.legend()
		fig.tight_layout()
		plt.show()

def main():
	htg = Histogram()
	htg.hogwarts_histogram()

if __name__ == "__main__":
    main()


    
    
