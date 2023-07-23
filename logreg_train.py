import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from gradient_descent import GradientDescent
ft_linear_regression.path.append('/mnt/nfs/homes/ymarcais/ft_linear_regression')

@dataclass
class Logreg_train:





def main():
    path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	unique_house_df = pd.DataFrame()  
	db = Describe()
    gd = GradientDescent()
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
	gd.my_linear_regression(normalized_numerical_data)

if __name__ == "__main__":
    main()