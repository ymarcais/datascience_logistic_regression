import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass
from describe import Describe
from sklearn.preprocessing import StandardScaler


@dataclass
class DataNormalize:
    path: str
    filename: str
    db: Describe

    # Get data
    def import_data(self):
        self.dataset = self.db.open_data(self.path)
        return self.dataset

    # Clean data
    def clean_data(self):
        self.db.distribution_NaN(self.dataset)
        self.db.del_NaN_column(self.dataset)
        cleaned_dataset = self.db.del_NaN_row(self.dataset)
        return cleaned_dataset

    # Separate copy of the numerical columns
    def separate_numerical(self, cleaned_dataset):
        numerical_columns = cleaned_dataset.select_dtypes(include=['int', 'float']).columns
        self.numerical_data = cleaned_dataset[numerical_columns].copy()

    # StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1
    # fit() calculates the mean and the std, transform() applies to the data
    def normalizer(self):
        scaler = StandardScaler()
        scaler.fit(self.numerical_data)
        self.normalized_numerical_data = scaler.transform(self.numerical_data)

    # Create a new DataFrame with the normalized numerical data and original index from cleaned_dataset
    def index_to_normalized_data(self):
        df_normalized = pd.DataFrame(
            self.normalized_numerical_data,
            columns=self.numerical_data.columns,
            index=self.numerical_data.index
        )
        return df_normalized


@dataclass
class Statistical:
    db: Describe
    dn: DataNormalize
    unique_house_df: pd.DataFrame

    # Calculate mean for each student
    def note_student_mean(self):
        count = self.db.count(self.dn.normalized_numerical_data)
        df_mean = pd.DataFrame()
        for index, row in self.dn.normalized_numerical_data.iterrows():
            total = 0
            for col in self.dn.normalized_numerical_data.columns:
                total += row[col]
            mean = total / count
            df_mean.loc[index, 'student mean'] = mean
        return df_mean

    # Add house name of house to df_mean dataframe (using index as reference)
    def add_house_name(self, cleaned_dataset, df_mean):
        df_mean['Hogwarts house'] = cleaned_dataset.loc[df_mean.index, 'Hogwarts house']
        return df_mean

    # Create unique house name
    def unique_house(self, df_mean):
        unique_house_count = df_mean['Hogwarts house'].nunique()
        self.unique_house_df = pd.DataFrame({'Hogwarts house': df_mean['Hogwarts house'].unique()})
        return unique_house_count

    # Count number of students per house
    def count_student_per_house(self, df_mean):
        house_counts = df_mean['Hogwarts house'].value_counts().reset_index()
        house_counts.columns = ['Hogwarts house', 'count']
        self.unique_house_df = pd.merge(self.unique_house_df, house_counts, on='Hogwarts house', how='left')
        return self.unique_house_df

    # Sum all the notes of one house
    def sum_student_notes_per_house(self, df_mean):
        sum_notes = df_mean.groupby('Hogwarts house')['student mean'].sum().reset_index()
        sum_notes.columns = ['Hogwarts house', 'sum notes']
        self.unique_house_df = pd.merge(self.unique_house_df, sum_notes, on='Hogwarts house', how='left')
        return self.unique_house_df

    # Calculate mean for each house
    def house_mean(self):
        self.unique_house_df['house mean'] = self.unique_house_df['sum notes'] / self.unique_house_df['count']
        return self.unique_house_df

    def calculate_std(self, df_mean):
        for house in self.unique_house_df['Hogwarts house']:
            students_mean = df_mean[df_mean['Hogwarts house'] == house]['student mean']
            house_counts = self.unique_house_df[self.unique_house_df['Hogwarts house'] == house]['count']
            house_mean = self.unique_house_df[self.unique_house_df['Hogwarts house'] == house]['house mean']
            std = (np.sum((students_mean - house_mean) ** 2) / (house_counts - 1)) ** 0.5
            self.unique_house_df.loc[self.unique_house_df['Hogwarts house'] == house, 'std'] = std
        return self.unique_house_df


@dataclass
class Histogram:
    data_normalizer: DataNormalize
    st: Statistical

    def hogwarts_histogram(self):
        std_values = self.st.unique_house_df['std']
        n_bins = len(std_values)  # Number of bins equal to the number of columns

        fig, ax = plt.subplots()  # Create figure and axis objects

        for i, column in enumerate(self.data_normalizer.normalized_numerical_data.columns):
            column_values = self.data_normalizer.normalized_numerical_data[column]
            bin_range = (i, i + 1)  # Range for the bin positions on the x-axis

            # Plot histogram bars using plt.bar
            hist, bins = np.histogram(column_values, bins=n_bins)
            ax.bar(bin_range[0], std_values[i], width=1, alpha=0.7, label=column)

        ax.set_xlabel(' ')
        ax.set_ylabel('Value')
        ax.set_title('{}'.format(self.data_normalizer.filename))
        ax.set_xticks(range(len(std_values)))  # Set the x-ticks to correspond to the column index
        ax.set_xticklabels(std_values.index, rotation=45, ha='right')  # Set the x-tick labels to the column names
        ax.set_ylim(0, np.max(std_values))  # Set the y-axis limits based on the maximum value in std_values
        ax.set_yscale('log', base=1.01)

        fig.tight_layout()
        plt.show()


def main():
    path = "datasets/dataset_train.csv"
    filename = path.split("/")[-1]
    unique_house_df = pd.DataFrame()
    db = Describe()
    dn = DataNormalize(path, filename, db)
    cleaned_dataset = dn.clean_data()
    dn.separate_numerical(cleaned_dataset)
    dn.normalizer()
    df_normalized = dn.index_to_normalized_data()
    st = Statistical(db, dn, unique_house_df)
    df_mean = st.note_student_mean()
    df_mean = st.add_house_name(cleaned_dataset, df_mean)
    unique_house_count = st.unique_house(df_mean)
    unique_house_df = st.count_student_per_house(df_mean)
    unique_house_df = st.sum_student_notes_per_house(df_mean)
    unique_house_df = st.house_mean()
    unique_house_df = st.calculate_std(df_mean)
    htg = Histogram(dn, st)
    htg.hogwarts_histogram()


if __name__ == "__main__":
    main()
