import json
import pandas as pd
import time
from sklearn.utils import shuffle

from module.data_processing.data_preparation import DataPreparation
from module.data_processing.data_preprocessing import DataPreprocessing
from module.data_processing.data_postprocessing import DataPostprocessing
from module.model.synthesizers.synthesizer import CTABGANSynthesizer


class CTABGAN:
    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical
    distribution
    6) integer_columns -> list of numeric column names without floating numbers
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data
    3) generate_samples() -> returns a generated and post-processed synthetic dataframe with the same size and format
    as per the input data
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
                 processed_csv_path: str,
                 # chunk_size: int,
                 # chunk_csv_path: str
                 ):

        self.lower_bounds = None
        self.column_types = None
        self.columns = None
        self.label_encoder_list = None
        self.categorical_columns_minor_terms = None
        self.sample_length = 0
        self.processed_csv_path = processed_csv_path
        self.raw_csv_path = raw_csv_path
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.target = target
        self.num_epochs = num_epochs
        # self.chunk_size = chunk_size
        self.synthesizer = CTABGANSynthesizer()
        self.training_time = 0
        # self.chunk_csv_file = chunk_csv_path

    def fit(self):
        start_time = time.time()
        # total_size = sum(1 for line in open('your_dataset.csv')) - 1
        # num_chunks = total_size // self.chunk_size + 1
        # print(f'We have {num_chunks} chunks and {self.num_epochs} epochs in total.')
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch + 1}')

            # chunk_indices = list(range(num_chunks))
            # chunk_indices = shuffle(chunk_indices)
            # for i, chunk_idx in enumerate(chunk_indices):
            #     print(f'Training {i}-th trunk in epoch {epoch + 1}.')
            #     current_chunk_size = min(self.chunk_size, total_size - chunk_idx * self.chunk_size)
            #     chunk = pd.read_csv('your_dataset.csv', skiprows=chunk_idx * self.chunk_size + 1,
            #                         nrows=current_chunk_size, header=None)
            #     self.data_preparation = DataPreparation(raw_df=chunk,
            #                                             categorical_columns=self.categorical_columns,
            #                                             log_applied_columns=self.log_columns,
            #                                             mixed_columns=self.mixed_columns,
            #                                             integer_column=self.integer_columns,
            #                                             target=self.target,
            #                                             test_ratio=self.test_ratio,
            #                                             # chunk_csv_path=self.chunk_csv_file
            #                                             )
            #     self.synthesizer.fit(train_data=self.data_preparation.data,
            #                          categorical=self.data_preparation.get_column_type["categorical"],
            #                          mixed=self.data_preparation.get_column_type["mixed"],
            #                          type=self.target)
            #     self.categorical_columns = self.data_preparation.categorical_column
            DataPreprocessing(raw_csv_path=self.raw_csv_path,
                              categorical_column=self.categorical_columns,
                              mixed_columns=self.mixed_columns,
                              integer_columns=self.integer_columns,
                              processed_csv_path=self.processed_csv_path)

            data = pd.read_csv(self.processed_csv_path)
            with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/integer_columns.json', 'r') \
                    as integer_columns_file:
                self.integer_columns = json.load(integer_columns_file)
            with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/categorical_columns_minor_terms' +
                      'json', 'r') as categorical_columns_minor_terms_file:
                self.categorical_columns_minor_terms = json.load(categorical_columns_minor_terms_file)
            self.sample_length = len(data)

            DataPreparation(raw_df=data,
                            categorical_columns=self.categorical_columns,
                            log_applied_columns=self.log_columns,
                            mixed_columns=self.mixed_columns,
                            integer_column=self.integer_columns,
                            target=self.target,
                            test_ratio=self.test_ratio,
                            # chunk_csv_path=self.chunk_csv_file
                            )
            del data

            data = pd.read_csv('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/training_data.csv')
            with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/column_types.json', 'r') \
                    as column_types_file:
                self.column_types = json.load(column_types_file)
            with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/lower_bounds.json', 'r') \
                    as lower_bounds_file:
                self.lower_bounds = json.load(lower_bounds_file)
            with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/label_encoder_list.json', 'r') \
                    as label_encoder_list_file:
                self.label_encoder_list = json.load(label_encoder_list_file)
            with open('/content/drive/MyDrive/CTABGANforClickThrough/data_processing/columns.json', 'r') \
                    as columns_file:
                self.columns = json.load(columns_file)

            self.synthesizer.fit(train_data=data,
                                 categorical=self.column_types["categorical"],
                                 mixed=self.column_types["mixed"],
                                 types=self.target)
        end_time = time.time()
        self.training_time = end_time - start_time
        self.print_training_time()

    def print_training_time(self) -> None:
        print(f'Finished training in {self.training_time} seconds.')

    def generate_samples(self, eps=1) -> tuple[pd.DataFrame, pd.DataFrame]:
        sample = self.synthesizer.sample(self.sample_length / 100)
        postprocessed_data = DataPostprocessing(generated_data=sample,
                                                training_data_columns=self.columns,
                                                categorical_columns=self.categorical_columns,
                                                log_applied_columns=self.log_columns,
                                                mixed_columns=self.mixed_columns,
                                                integer_column=self.integer_columns,
                                                categorical_columns_minor_terms=self.categorical_columns_minor_terms,
                                                label_encoder_list=self.label_encoder_list,
                                                lower_bounds=self.lower_bounds)
        sample_df1 = postprocessed_data.inverse_most(eps)
        sample_df2 = postprocessed_data.inverse_all(eps)
        return sample_df1, sample_df2
