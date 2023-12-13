import numpy as np
import pandas as pd
import json

from sklearn import preprocessing
from sklearn import model_selection


class DataPreparation:

    """
    Data preparation class for modifying the datasets to standard forms.
    """

    def __init__(self,
                 raw_df: pd.DataFrame,
                 categorical_columns: list,
                 log_applied_columns: list,
                 mixed_columns: dict,
                 integer_column: list,
                 target: dict,
                 test_ratio: float,
                 root_path: str
                 ):

        self.data = raw_df
        self.categorical_column = categorical_columns
        self.log_column = log_applied_columns
        self.mixed_column = mixed_columns
        self.integer_column = integer_column
        self.target = target
        self.test_ratio = test_ratio
        self.root_path = root_path

        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []
        self.categorical_columns_minor_terms = {}
        self.training_data = pd.DataFrame()

    def _prepare(self):
        self.split_data()
        self.transform_skewed_columns()
        self.encode_categorical_columns()
        self.write_training_data()
        self.write_columns()
        self.write_column_types()
        self.write_label_encoder_list()
        self.write_lower_bounds()

    def split_data(self):
        target_column = list(self.target.values())[0]

        X_real = self.data.drop(columns=[target_column])
        y_real = self.data[target_column]

        X_train, _, y_train, _ = model_selection.train_test_split(X_real, y_real, test_size=self.test_ratio,
                                                                  shuffle=True, stratify=y_real, random_state=42)
        X_train[target_column] = y_train
        self.training_data = X_train

    def transform_skewed_columns(self):
        if self.log_column:
            for log_column in self.log_column:
                eps = 1
                lower = np.min(self.training_data.loc[self.training_data[log_column] != -9999999][log_column].values)
                self.lower_bounds[log_column] = lower

                if lower > 0:
                    self.training_data[log_column] = \
                        self.training_data[log_column].apply(
                            lambda x: np.log(x) if x != -9999999 else -9999999)
                elif lower == 0:
                    self.training_data[log_column] = \
                        self.training_data[log_column].apply(
                            lambda x: np.log(x + eps) if x != -9999999 else -9999999)
                else:
                    self.training_data[log_column] = \
                        self.training_data[log_column].apply(
                            lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999)

    def encode_categorical_columns(self):
        for col_index, col in enumerate(self.training_data.columns):
            if col in self.categorical_column:
                label_encoder = preprocessing.LabelEncoder()
                self.training_data[col] = self.training_data[col].astype(str)
                label_encoder.fit(self.training_data[col])
                current_label_encoder = dict()
                current_label_encoder['column'] = col
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(self.training_data[col])
                self.training_data[col] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(col_index)
            elif col in self.mixed_column:
                self.column_types["mixed"][col_index] = self.mixed_column[col]

    def write_training_data(self):
        self.data.to_csv(self.root_path + '/data_processing/training_data.csv')

    def write_column_types(self):
        with open(self.root_path + '/data_processing/column_types.json', 'w') \
                as column_types_file:
            json.dump(self.column_types, column_types_file)

    def write_columns(self):
        with open(self.root_path + '/data_processing/columns.json', 'w') as columns_file:
            json.dump(self.training_data.columns, columns_file)

    def write_label_encoder_list(self):
        with open(self.root_path + '/data_processing/label_encoder_list.json', 'w') as label_encoder_list_file:
            json.dump(self.label_encoder_list, label_encoder_list_file)

    def write_lower_bounds(self):
        with open(self.root_path + '/data_processing/lower_bounds.json', 'w') as lower_bounds_file:
            json.dump(self.lower_bounds, lower_bounds_file)
