"""
Classes and functions to facilitate obtaining the traces and trace summaries,
interpreting the results, and generating the relevant plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, anderson

from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score


def generate_tracesum_plot(features, noise_amts, tracesums, title, v='sd', ax=None):
    dff = {}
    for f in features:
        dff[f] = [x.loc[f, v] for x in tracesums]
    df = pd.DataFrame.from_dict(dff)
    df.index = noise_amts
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    sns.lineplot(df, markers=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('noise ratio')
    ax.set_ylabel(v)
    return fig, ax


def generate_kstest_plot(features, noise_amts, traces, title):
    dff = {}
    for f in features:
        dff[f] = [kstest(x[f], 'norm').pvalue for x in traces]
    df = pd.DataFrame.from_dict(dff)
    df.index = noise_amts

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    sns.lineplot(df, markers=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('noise ratio')
    ax.set_ylabel('KS test p-value')
    return fig, ax


def generate_anderson_plot(features, noise_amts, traces, title):
    dff = {}
    for f in features:
        dff[f] = [anderson(x[f], 'norm').statistic for x in traces]
    df = pd.DataFrame.from_dict(dff)
    df.index = noise_amts

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    sns.lineplot(df, markers=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('noise ratio')
    ax.set_ylabel('Anderson test statistic')
    return fig, ax


def generate_multiple_roc(y_tests, y_scores, line_kwargs, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    for y_test, y_score, line_kw in zip(y_tests, y_scores, line_kwargs):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        line_kw['label'] = line_kw['label'].format(roc_auc)

        ax.plot(
            fpr,
            tpr,
            **line_kw
        )

    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig, ax
