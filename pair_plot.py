from histogram import Data_normalize
from histogram import Statistiscal
from describe import Describe
from dataclasses import dataclass
from scatter_plot import R_correlation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Pairplot_graph:

	def data_(self, path):
		db = Describe()
		dn = Data_normalize(path, db)
		st = Statistiscal(db, dn)
		dataset = dn.import_data()
		cleaned_dataset = dn.clean_data(dataset)
		numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
		numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
		normalized_numerical_data = dn.normalizator(numerical_data)
		numerical_data_indexed = dn.index_to_normalized_data(normalized_numerical_data, numerical_columns, cleaned_dataset)
		dataframe = pd.DataFrame()
		dataframe = st.add_house_name(numerical_data_indexed, cleaned_dataset)
		return dataframe
		
	
	def pairplot_chart(self, dataframe):
		sns.set(style="ticks")
		sns.pairplot(dataframe.iloc[:, [2, 3,4, 0]], hue="Hogwarts House", palette="Set1")
		plt.show()


def main():
	path = "datasets/dataset_train.csv"
	db = Describe()
	numerical_data = []
	dn = Data_normalize(path, db)
	st = Statistiscal(db, dn)
	pg = Pairplot_graph()
	dataframe = pg.data_(path)
	pg.pairplot_chart(dataframe)


if __name__ == "__main__":
    main()



