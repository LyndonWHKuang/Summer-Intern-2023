import numpy as np
import pandas as pd


class DataPostprocessing:

    """
    Transform the generated data to original form.
    """

    def __init__(self,
                 generated_data: pd.DataFrame,
                 training_data_columns: pd.Series,
                 categorical_columns: list,
                 log_applied_columns: list,
                 mixed_columns: dict,
                 integer_column: list,
                 categorical_columns_minor_terms: dict,
                 label_encoder_list: list,
                 lower_bounds: dict,
                 eps: int = 1
                 ):

        self.data = generated_data
        self.training_data_columns = training_data_columns
        self.categorical_columns = categorical_columns
        self.log_columns = log_applied_columns
        self.mixed_columns = mixed_columns
        self.integer_column = integer_column
        self.categorical_columns_minor_terms = categorical_columns_minor_terms
        self.label_encoder_list = label_encoder_list
        self.lower_bounds = lower_bounds
        self.data_sample = pd.DataFrame()

    def inverse_most(self, eps=1):
        self.inverse_encoding_categorical_columns()
        self.inverse_log_transform(eps)
        self.round_integer()
        self.fill_categorical_minor_terms()
        self.recover_missing_value()

    def inverse_all(self, eps=1):
        self.inverse_most(eps)
        self.merge_date()

    def inverse_encoding_categorical_columns(self):
        self.data_sample = pd.DataFrame(self.data, columns=self.training_data_columns)
        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            col = self.label_encoder_list[i]["column"]
            self.data_sample[col] = le.inverse_transform(self.data_sample[col])

    def inverse_log_transform(self, eps=1):
        if self.log_columns:
            for column in self.log_columns:
                lower_bound = self.lower_bounds[column]

                if lower_bound > 0:
                    self.data_sample[column].apply(lambda x: np.exp(x) if x != -9999999 else -9999999)
                elif lower_bound == 0:
                    self.data_sample[column] = self.data_sample[column].apply(
                        lambda x: np.ceil(np.exp(x) - eps) if ((x != -9999999) & ((np.exp(x) - eps) < 0)) else (
                            np.exp(x) - eps
                            if x != -9999999 else -9999999))
                else:
                    self.data_sample[column] = self.data_sample[column].apply(
                        lambda x: np.exp(x) - eps + lower_bound if x != -9999999 else -9999999)

    def round_integer(self):
        if self.integer_column:
            for column in self.integer_column:
                self.data_sample[column] = (np.round(self.data_sample[column].values)).astype(int)

    def fill_categorical_minor_terms(self):
        for column in self.data_sample:
            if column in self.categorical_columns_minor_terms.keys():
                mask = self.data_sample[column] == "others"
                replace_count = mask.sum()
                probabilities = self.categorical_columns_minor_terms.get(column, {})
                replacement = np.random.choice(list(probabilities.keys()), replace_count,
                                               p=list(probabilities.values()))
                self.data_sample.loc[mask, column] = replacement

    def recover_missing_value(self):
        self.data_sample.replace(-9999999, np.nan, inplace=True)
        self.data_sample.replace('empty', np.nan, inplace=True)

    def merge_date(self):
        self.data_sample['hour'] = self.data_sample.apply(
            lambda row: f"{row['Year']:0>2}{row['Month']:0>2}{row['Day']:0>2}{row['Hour']:0>2}", axis=1)
        self.data_sample.drop('Year', axis=1)
        self.data_sample.drop('Month', axis=1)
        self.data_sample.drop('Day', axis=1)
        self.data_sample.drop('Hour', axis=1)
