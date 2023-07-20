# To check feature similarities we'll use Pearson's correlation: 
# r = (Σ((X - X̄) * (Y - Ȳ))) / sqrt(Σ((X - X̄)^2) * Σ((Y - Ȳ)^2))

from	histogram	import	Data_normalize as dn
fron	describe	import	Describe


class Pearson_correlation:

	def size_(df_normalized)
		row_count = 0
			for _, _ in df_normalized.iterrows():
				row_count += 1
		return row_count
    
    def iter_rows(feature_1, feature_2):g
        for 

def	main():
	path = "datasets/dataset_train.csv"
	filename = path.split("/")[-1]
	db = Describe()
	dn = Data_normalize(path, filename, db)
	dataset = dn.import_data()
	cleaned_dataset = dn.clean_data()
	numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
	numerical_data = dn.separate_numerical(cleaned_dataset, numerical_columns)
	normalized_numerical_data = dn.normalizator(numerical_data)
	df_normalized = dn.index_to_normalized_data(normalized_numerical_data, numerical_columns, cleaned_dataset)

if __name__ == "__main__"
	main()