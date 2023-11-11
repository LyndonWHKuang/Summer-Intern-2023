import numpy as np
import pandas as pd


class DataPreprocessing:

    """
    Preprocess the raw data csv to for chunk training.
    """

    def __init__(self,
                 raw_csv_path: str,
                 categorical_column: list,
                 mixed_columns: dict,
                 processed_csv_path: str
                 ):
        self.raw_csv_path = raw_csv_path
        self.categorical_column = categorical_column
        self.mixed_column = mixed_columns
        self.processed_csv_path = processed_csv_path

        self.categorical_columns_minor_terms = {}
        self.data = pd.DataFrame()

    def preprocess(self):
        self.data = pd.read_csv(self.raw_csv_path)
        self.data_transformation()
        self.handle_missing_values()
        self.handle_superfluous_categories()
        self.write_processed_data()

    def data_transformation(self):
        new_data = self.data.copy()  # copy the dataframe to avoid modifying the original one
        new_data = new_data.drop('id', axis=1)
        new_data['hour'] = new_data['hour'].astype(str)
        new_data['Year'] = new_data['hour'].apply(lambda x: "20" + x[:2])
        new_data['Month'] = new_data['hour'].apply(lambda x: x[2:4])
        new_data['Day'] = new_data['hour'].apply(lambda x: x[4:6])
        new_data['Hour'] = new_data['hour'].apply(lambda x: x[6:])
        new_data[['Year', 'Month', 'Day', 'Hour']] = new_data[['Year', 'Month', 'Day', 'Hour']].astype(int)
        new_data = new_data.drop('hour', axis=1)

        # Save processed dataframe back to the original csv file
        # new_data.to_csv(self.chunk_csv_path, index=False)
        self.data = new_data
        self.mixed_column['Hour'] = []
        self.mixed_column['Year'] = []
        self.mixed_column['Month'] = []
        self.mixed_column['Day'] = []

    def handle_missing_values(self):
        self.data = self.data.replace(r' ', np.nan)
        self.data = self.data.fillna('empty')

        all_columns = set(self.data.columns)
        irrelevant_missing_columns = set(self.categorical_column)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for column in relevant_missing_columns:
            if column in list(self.mixed_column.keys()):
                if "empty" in list(self.data[column].values):
                    self.data[column] = self.data[column].apply(
                        lambda x: -9999999 if x == "empty" else x)
                    self.mixed_column[column].append(-9999999)
            else:
                if "empty" in list(self.data[column].values):
                    self.data[column] = self.data[column].apply(
                        lambda x: -9999999 if x == "empty" else x)
                    self.mixed_column[column] = [-9999999]

    def handle_superfluous_categories(self):
        processed_data = self.data
        data_length = self.data.shape[0]
        for col in processed_data:
            if col == 'id':
                continue
            else:
                if col in self.categorical_column:
                    all_minor_terms_sum = 0
                    this_column = self.data[col].value_counts()
                    if len(this_column) > 25:
                        self.categorical_columns_minor_terms[col] = {}
                        for idx in this_column.index:
                            if this_column[idx] < 0.05 * data_length and idx != -9999999:
                                all_minor_terms_sum += this_column[idx]
                                self.categorical_columns_minor_terms[col][idx] = this_column[idx]
                                processed_data[col].apply(lambda x: "others")
                        for idx in self.categorical_columns_minor_terms[col]:
                            self.categorical_columns_minor_terms[col][idx] /= all_minor_terms_sum
        self.data = processed_data

    def write_processed_data(self):
        self.data.to_csv(self.processed_csv_path)

    def get_mixed_columns(self):
        return self.mixed_column

    def get_categorical_columns_minor_terms(self):
        return self.categorical_columns_minor_terms
