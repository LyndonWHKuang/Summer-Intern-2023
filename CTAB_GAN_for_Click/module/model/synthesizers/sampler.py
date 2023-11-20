import numpy as np
from typing import List, Tuple
from module.Utils import random_choice_prob_index_sampling


class ConditionalVectorSampler:
    """
    This class is responsible for sampling conditional vectors to be supplied to the generator.

    Attributes:
    1) categories: list containing an index of highlighted categories in their corresponding one-hot-encoded
    representations.
    2) intervals: an array holding the respective one-hot-encoding starting positions and sizes.
    3) n_cols: total number of one-hot-encoding representations.
    4) n_options: total number of distinct categories across all one-hot-encoding representations.
    5) log_prob_distributions: list containing log of probability mass distribution of categories within their
    respective one-hot-encoding representations.
    6) prob_distributions: list containing probability mass distribution of categories within their respective
    one-hot-encoding representations.
    """

    def __init__(self, data: np.ndarray, output_info: List[Tuple[int, str]]):
        self.categories = []
        self.intervals = np.array([])
        self.n_cols = 0
        self.n_options = 0
        self.log_prob_distributions = []
        self.prob_distributions = []

        self.compute_attributes(data, output_info)

    def compute_attributes(self, data: np.ndarray, output_info: List[Tuple[int, str]]):
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                self.categories.append(np.argmax(data[:, st:ed], axis=-1))
                self.intervals = np.append(self.intervals, (self.n_options, item[0]))
                self.n_cols += 1
                self.n_options += item[0]
                freq = np.sum(data[:, st:ed], axis=0)
                log_freq = np.log(freq + 1)
                log_pmf = log_freq / np.sum(log_freq)
                self.log_prob_distributions.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.prob_distributions.append(pmf)
                st = ed
        self.intervals = np.asarray(self.intervals)

    def sample_for_training(self, batch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.n_cols != 0:
            batch = batch
            vec = np.zeros((batch, self.n_options), dtype='float32')
            idx = np.random.choice(np.arange(self.n_cols), batch)
            mask = np.zeros((batch, self.n_cols), dtype='float32')
            mask[np.arange(batch), idx] = 1
            selected_categories = random_choice_prob_index_sampling(self.log_prob_distributions, idx)
            for i in np.arange(batch):
                vec[i, self.intervals[idx[i], 0] + selected_categories[i]] = 1

            return vec, mask, idx, selected_categories

    def sample_for_generation(self, batch: int) -> np.ndarray:
        if self.n_cols != 0:
            batch = batch
            vec = np.zeros((batch, self.n_options), dtype='float32')
            idx = np.random.choice(np.arange(self.n_cols), batch)
            selected_categories = random_choice_prob_index_sampling(self.prob_distributions, idx)
            for i in np.arange(batch):
                vec[i, self.intervals[idx[i], 0] + selected_categories[i]] = 1
            return vec


class RealDataSampler:
    """
    This class is used to sample the transformed real data according to the conditional vector.

    Attributes:
    1) data: real transformed input data.
    2) category_indices: stores the index values of data records corresponding to any given selected categories for
    all columns.
    3) data_size: size of the input data.

    Methods:
    1) __init__: initiates the sampler object and stores class attributes.
    2) sample_data: takes as input the number of rows to be sampled (n), chosen column (col),
                    and category within the column (opt) to sample real records accordingly.
    """

    def __init__(self, data: np.ndarray, output_info: List[Tuple[int, str]]):
        super(RealDataSampler, self).__init__()
        self.data = data
        self.model = []
        self.data_size = len(data)
        self.populate_category_indices(output_info)

    def populate_category_indices(self, output_info: List[Tuple[int, str]]):
        # counter to iterate through columns
        st = 0
        # iterating through column information
        for item in output_info:
            # ignoring numeric columns
            if item[1] == 'tanh':
                st += item[0]
                continue
            # storing indices of data records for all categories within one-hot-encoded representations
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                # iterating through each category within a one-hot-encoding
                for j in range(item[0]):
                    # storing the relevant indices of data records for the given categories
                    tmp.append(np.nonzero(self.data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample_data(self, n, col, opt):

        flag = False

        # if there are no one-hot-encoded representations, we may ignore sampling using a conditional vector
        if col is None:
            idx = np.random.choice(np.arange(self.data_size), n)
            return self.data[idx]

        # used to store relevant indices of data records based on selected category within a chosen one-hot-encoding
        idx = []

        # sampling a data record index randomly from all possible indices that meet the given criteria of the chosen
        # category and one-hot-encoding
        for c, o in zip(col, opt):
            if len(self.model[c][o]) != 0:
                idx.append(np.random.choice(self.model[c][o]))
                flag = True
            else:
                flag = False
        return self.data[idx], flag
