import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/mnt/nfs/homes/ymarcais/ft_linear_regression')
from gradient_descent import GradientDescent
from describe import Describe
from histogram import Data_normalize
from histogram import Statistiscal
from scatter_plot	import R_correlation

#@dataclass
#class Logreg_train:

def	main():
	path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	unique_house_df = pd.DataFrame()  
	db = Describe()
	thetas = np.array([0, 0]).reshape((-1, 1))
	thetas = thetas.astype('float64')
	max_iter = 50000
	alpha = np.array([0.001, 0.000340])
	alpha = alpha.astype('float64')
	gd = GradientDescent(thetas, alpha, max_iter)
	numerical_data = []
	dn = Data_normalize(path, filename, db)
	st = Statistiscal(db, dn, unique_house_df)
	rc = R_correlation(db, dn, r_array=[], df_normalized=[], count_rows = 0, last_r = 0)
	df_normalized = rc.get_normalized_data()
	dataset = dn.import_data()
	cleaned_dataset = dn.clean_data()
	numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
	numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
	normalized_numerical_data = dn.normalizator(numerical_data)
	#gd.my_linear_regression(normalized_numerical_data)
	print(numerical_data )

if __name__ == "__main__":
    main()