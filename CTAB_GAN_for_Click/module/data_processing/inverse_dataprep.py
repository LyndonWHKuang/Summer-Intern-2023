import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection


class InverseDataPreparation:

    """
    Transform the generated data to original form.
    """

    def __init__(self,
                 generated_data: pd.DataFrame,
                 categorical_columns: list,
                 log_applied_columns: list,
                 mixed_columns: dict,
                 integer_column: list,
                 target: dict,
                 categorical_columns_minor_terms: dict,
                 label_encoder_list: list,

                 ):
        self.data = generated_data
        self.categorical_columns = categorical_columns
        self.log_columns = log_applied_columns
        self.mixed_columns = mixed_columns
        self.integer_column = integer_column
        self.target = target
        self.categorical_columns_minor_terms = categorical_columns_minor_terms

