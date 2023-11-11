import numpy as py
import pandas as pd
import time
from sklearn.utils import shuffle

from module.data_processing.data_preparation import DataPreparation
from module.model.synthesizer import CTABGANSynthesizer


class CTABGAN():
    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution
    6) integer_columns -> list of numeric column names without floating numbers
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data
    3) generate_samples() -> returns a generated and post-processed sythetic dataframe with the same size and format as per the input data
    """

    def __init__(self,
                 raw_csv_path: str,
                 test_ratio: float,
                 categorical_columns: list,
                 log_columns: list,
                 mixed_columns: dict,
                 integer_columns: list,
                 target: dict,
                 num_epochs: int,
                 chunk_size: int,
                 chunk_csv_path: str):

        self.raw_csv_path = raw_csv_path
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.target = target
        self.num_epochs = num_epochs
        self.chunk_size = chunk_size
        self.synthesizer = CTABGANSynthesizer()
        self.data_preparation = None
        self.training_time = 0
        # self.chunk_csv_file = chunk_csv_path

    def fit(self):
        start_time = time.time()
        total_size = sum(1 for line in open('your_dataset.csv')) - 1
        num_chunks = total_size // self.chunk_size + 1
        print(f'We have {num_chunks} chunks and {self.num_epochs} epochs in total.')
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch + 1}')

            chunk_indices = list(range(num_chunks))
            chunk_indices = shuffle(chunk_indices)
            for i, chunk_idx in enumerate(chunk_indices):
                print(f'Training {i}-th trunk in epoch {epoch + 1}.')
                current_chunk_size = min(self.chunk_size, total_size - chunk_idx * self.chunk_size)
                chunk = pd.read_csv('your_dataset.csv', skiprows=chunk_idx * self.chunk_size + 1,
                                    nrows=current_chunk_size, header=None)
                self.data_preparation = DataPreparation(raw_df=chunk,
                                                        categorical_columns=self.categorical_columns,
                                                        log_applied_columns=self.log_columns,
                                                        mixed_columns=self.mixed_columns,
                                                        integer_column=self.integer_columns,
                                                        target=self.target,
                                                        test_ratio=self.test_ratio,
                                                        # chunk_csv_path=self.chunk_csv_file
                                                        )
                self.synthesizer.fit(train_data=self.data_preparation.data,
                                     categorical=self.data_preparation.get_column_type["categorical"],
                                     mixed=self.data_preparation.get_column_type["mixed"],
                                     type=self.target)
                self.categorical_columns = self.data_preparation.categorical_column
        end_time = time.time()
        self.training_time = end_time - start_time
        self.print_training_time()

    def print_training_time(self) -> None:
        print(f'Finished training in {self.training_time} seconds.')
