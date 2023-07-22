from histogram import Data_normalize
from histogram import Statistiscal
from describe import Describe
from dataclasses import dataclass
from scatter_plot import R_correlation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Pairplot_graph:

	def pairplot_chart(self, dataframe):
		#hue_columns = ['Divination', 'Muggle Studies', 'Ancient Runes']
		#sns.pairplot(selected_columns, hue = dataframe['Combined_Hue'])
		#plt.show()
		sns.set(style="ticks")
		sns.pairplot(dataframe.iloc[:, [3, 7, 13]], hue="Hogwarts House", palette="Set1")
		plt.show()
		
		
		'''g = sns.PairGrid(dataframe.iloc[:, 1:4])
		g.map_upper(sns.scatterplot)
		g.map_lower(sns.scatterplot)
		g.map_diag(sns.histplot, kde=True)
		plt.show()'''


def main():
	'''path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	db = Describe()
	dn = Data_normalize(path, filename, db)
	pg = Pairplot_graph()
	rc = R_correlation(db, dn, r_array=[], df_normalized=[], count_rows = 0, last_r = 0)
	df_normalized = rc.get_normalized_data()
	pg.pairplot_chart(df_normalized)'''

	path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	unique_house_df = pd.DataFrame()  
	db = Describe()
	numerical_data = []
	dn = Data_normalize(path, filename, db)
	st = Statistiscal(db, dn, unique_house_df)
	pg = Pairplot_graph()
	rc = R_correlation(db, dn, r_array=[], df_normalized=[], count_rows = 0, last_r = 0)
	df_normalized = rc.get_normalized_data()
	dataset = dn.import_data()
	cleaned_dataset = dn.clean_data()
	numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns


	numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
	normalized_numerical_data = dn.normalizator(numerical_data)
	numerical_data_indexed = dn.index_to_normalized_data(normalized_numerical_data, numerical_columns, cleaned_dataset)
	dataframe = st.add_house_name(numerical_data_indexed, cleaned_dataset)
	print(dataframe)
	pg.pairplot_chart(dataframe)


if __name__ == "__main__":
    main()



