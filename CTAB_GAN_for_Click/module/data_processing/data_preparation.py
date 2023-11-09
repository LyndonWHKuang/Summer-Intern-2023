import numpy as np
import pandas as pd

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
                 # chunk_csv_path: str
                 ):

        self.data = raw_df
        self.categorical_column = categorical_columns
        self.log_column = log_applied_columns
        self.mixed_column = mixed_columns
        self.integer_column = integer_column
        self.target = target
        self.test_ratio = test_ratio
        # self.chunk_csv_path = chunk_csv_path

        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []
        self.categorical_columns_minor_terms = {}
        self.training_data = pd.DataFrame()

    def _prepare(self):
        self.data_preprocessing()
        self.training_data = self.split_data()
        self.handle_missing_values()

    def data_preprocessing(self):
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

    def split_data(self):
        target_column = list(self.target.values())[0]

        X_real = self.data.drop(columns=[target_column])
        y_real = self.data[target_column]

        X_train, _, y_train, _ = model_selection.train_test_split(X_real, y_real, test_size=self.test_ratio,
                                                                  shuffle=True, stratify=y_real, random_state=42)
        X_train[target_column] = y_train
        self.training_data = X_train

    def handle_missing_values(self):
        self.training_data = self.training_data.replace(r' ', np.nan)
        self.training_data = self.training_data.fillna('empty')

        all_columns = set(self.training_data.columns)
        irrelevant_missing_columns = set(self.categorical_column)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for column in relevant_missing_columns:
            if column in list(self.mixed_column.keys()):
                if "empty" in list(self.training_data[column].values):
                    self.training_data[column] = self.training_data[column].apply(
                        lambda x: -9999999 if x == "empty" else x)
                    self.mixed_column[column].append(-9999999)
            else:
                if "empty" in list(self.training_data[column].values):
                    self.training_data[column] = self.training_data[column].apply(
                        lambda x: -9999999 if x == "empty" else x)
                    self.mixed_column[column] = [-9999999]

    def handle_superfluous_categories(self):
        processed_data = self.training_data
        data_length = self.training_data.shape[0]
        for col in processed_data:
            if col == 'id':
                continue
            else:
                if col in self.categorical_column:
                    all_minor_terms_sum = 0
                    this_column = self.training_data[col].value_counts()
                    if len(this_column) > 25:
                        self.categorical_columns_minor_terms[col] = {}
                        for idx in this_column.index:
                            if this_column[idx] < 0.05 * data_length and idx != -9999999:
                                all_minor_terms_sum += this_column[idx]
                                self.categorical_columns_minor_terms[col][idx] = this_column[idx]
                                processed_data[col].apply(lambda x: "others")
                        for idx in self.categorical_columns_minor_terms[col]:
                            self.categorical_columns_minor_terms[col][idx] /= all_minor_terms_sum
        self.training_data = processed_data

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
            if col in self.categorical_columncd:
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
                self.column_types["mixed"][col_index]= self.mixed_column[col]