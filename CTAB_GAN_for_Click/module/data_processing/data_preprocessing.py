import numpy as np
import pandas as pd
import json


class DataPreprocessing:
    """
    Preprocess the raw data csv to for chunk training.
    """

    def __init__(self,
                 raw_csv_path: str,
                 categorical_column: list,
                 mixed_columns: dict,
                 integer_columns: list,
                 processed_csv_path: str
                 ):
        self.raw_csv_path = raw_csv_path
        self.categorical_column = categorical_column
        self.mixed_column = mixed_columns
        self.integer_column = integer_columns
        self.processed_csv_path = processed_csv_path

        self.categorical_columns_minor_terms = {}
        self.data = pd.DataFrame()
        self.preprocess()

    def preprocess(self):
        self.data = pd.read_csv(self.raw_csv_path)
        print("Finished reading")
        print(self.data.head(5))
        self.data_transformation()
        self.handle_missing_values()
        self.handle_superfluous_categories()
        self.write_processed_data()
        self.write_integer_columns()
        self.write_categorical_columns_minor_terms()

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
        self.integer_column.append('Hour')
        self.integer_column.append('Year')
        self.integer_column.append('Month')
        self.integer_column.append('Day')

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
        data_length = self.data.shape[0]
        for col in self.data.columns:
            if col == 'id' or col not in self.categorical_column:
                continue

            value_counts = self.data[col].value_counts()
            if len(value_counts) > 25:
                self.categorical_columns_minor_terms[col] = {}
                minor_terms_mask = (value_counts < 0.05 * data_length) & (value_counts.index != -9999999)

                # Calculate the sum of all minor terms
                all_minor_terms_sum = value_counts[minor_terms_mask].sum()

                # Store the proportion of each minor term
                for term in value_counts[minor_terms_mask].index:
                    self.categorical_columns_minor_terms[col][term] = value_counts[term] / all_minor_terms_sum

                # Update the DataFrame to combine minor terms into "others"
                minor_terms_set = set(value_counts[minor_terms_mask].index)
                self.data[col] = self.data[col].apply(lambda x: "others" if x in minor_terms_set else x)

    def write_processed_data(self):
        self.data.to_csv(self.processed_csv_path)

    def write_integer_columns(self):
        with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/integer_columns.json', 'w') \
                as integer_columns_file:
            json.dump(self.integer_column, integer_columns_file)

    def write_categorical_columns_minor_terms(self):
        with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/categorical_columns_minor_terms.json',
                  'w') as categorical_columns_minor_terms_file:
            json.dump(self.categorical_columns_minor_terms, categorical_columns_minor_terms_file)
