"""
Classes and functions to facilitate creating the Bayesian Logistic Regression
model, training it, and performing inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score

import pymc3 as pm
from theano import shared


class Model:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass


class LRModel(Model):
    """
    Logistic Regression
    """
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.features = x.columns.values
        self.model.fit(X, y)
        return None

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def get_coefficients(self):
        coeffs = self.model.coef_
        df_coef = pd.DataFrame.from_dict({
            'feature': self.features,
            'coef': coeffs[0,:]
        }).sort_values(by='coef')

        return df_coef


class BLRModel(Model):
    """
    Bayesian Logistic Regression
    """
    def __init__(self, X, y, random_seed=100):
        self.random_seed = random_seed
        self.features = x.columns.values

        # data has to be provided prior to training
        with pm.Model() as log_model:
            lr_input = pm.Data("lr_input", X)
            lr_output = pm.Data("lr_output", y)

            # Weights from input to output
            lr_weights = pm.Normal(
                "lr_weights",
                0,
                sigma=1,
                shape=(X.shape[1],),
                testval=np.random.randn(X.shape[1]).astype(float)
            )

            # Build logistic model using sigmoid activation function
            act_out = pm.math.sigmoid(pm.math.dot(lr_input, lr_weights))

            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                act_out,
                observed=lr_output,
                total_size=y.shape[0],  # IMPORTANT for minibatches
            )

        self.model = log_model

    def fit(self, X=None, y=None, burnin=500, samples=500):
        # data was loaded prior
        with self.model:
            self.trace = pm.sample(samples, tune=burnin, init="adapt_diag", random_seed=self.random_seed)

        return self.trace

    def predict_proba(self, X, samples=100):
        with self.model:
            self.model['lr_input'].set_value(X)
            ppc = pm.sample_posterior_predictive(self.trace,
                                                model=self.model,
                                                samples=samples)

        return ppc['out'].mean(axis=0)

    def get_raw_trace(self):
        arr = np.vstack([t['lr_weights'] for t in self.trace])
        return pd.DataFrame(arr, columns=self.features)

    def get_trace_summary(self):
        return az.summary(self.trace)

    def get_model_diagram(self):
        return pm.model_graph.model_to_graphviz(model=self.model)