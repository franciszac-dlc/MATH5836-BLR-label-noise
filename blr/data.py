"""
Classes and functions to facilitate the generation of the data set, including
the type and amount of label noise to introduce.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


class Dataset:
    def __init__(self, X, y, noise_type=None, random_seed=100, test_split=0.4):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.noise_type = noise_type
        self.test_split = test_split
        self.X = X
        self.y = y

    def generate_split(self, noise_amt=0.0):
        # generate base splits
        # scaler = ColumnTransformer([('minmax-scaler', 
        #                             MinMaxScaler(),
        #                             ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
        #                             )], remainder='passthrough')
        scaler = StandardScaler()

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            self.X,
            self.y,
            random_state=self.random_seed)

        # Gelman recommends a std. dev. of 0.5
        X_train = scaler.fit_transform(X_train_raw) * 0.5
        X_test = scaler.transform(X_test_raw) * 0.5
        
        # noise as necessary
        if self.noise_type is None:
            return X_train, X_test, y_train, y_test

        randmask = self.rng.random(y_train.shape)
        
        if self.noise_type == 'uniform':
            # x% of datapoints will have their label flipped, regardless of label.
            y_train[randmask < noise_amt] = 1 - y_train[randmask < noise_amt]
        elif self.noise_type == 'pu':
            # x% of positives will have their label flipped. Negatives will be untouched.
            y_train[randmask < noise_amt] = 0

        # NOTE: y_test is untouched            
        return X_train, X_test, y_train, y_test

    def generate_splits_from_parameters(self, noise_amts=[0.0]):
        for na in noise_amts:
            yield self.generate_split(na)