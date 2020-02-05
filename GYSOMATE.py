"""
functions used for predict_conso.py
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-02-21>'
__author__ = "ChaoTANG@univ-reunion.fr"

import re
import pytz
import seaborn
import pyarrow.parquet as pq
from socket import *
from pysolar.solar import get_azimuth, get_altitude
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from dateutil import tz
import sys
from mpl_toolkits import mplot3d
import glob
import time
import click
import datetime
import pickle
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import metrics
from subprocess import call
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.externals import joblib
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import pdist
from pysolar.solar import get_altitude
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier, plot_importance

from pandas.plotting import register_matplotlib_converters
import urllib

register_matplotlib_converters()

os.environ['KMPDUPLICATELIB_OK'] = 'True'
# xgboost OMP: Error....Initializing libiomp5.dylib...

# DIR = '/etc/telegraf/execs/Prediction_PV'
DIR = '/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Prediction_PV'


def day_2_datetime(day, freq, periods):
    """ day is one (a list) of datetime.date.

    same as :
            df[df.index.month.isin([7,1,2,3]) & df.index.day.isin([1,20])]

    return a ndarray of datetime.datetime object:
                    output
    possible inputs:
                    1) pd.to_datetime('2017-07-08')
                    2) list(df.index.date[1:20])
                    3) list([pd.to_datetime(x) for x in ['2017-08-07', '2017-09-09']])

    then, select df rows by
                    df.loc[output]
                    """

    # 's' : second
    # 'min' : minute
    # 'H' : hour
    # 'D' : day
    # 'w' : week
    # 'm' : month

    if type(day) == list:
        length_day = len(day)
        index = pd.date_range(day[0], periods=periods, freq=freq)
        for i in range(1, length_day):
            # print(i, length_day)
            index = index.append(pd.date_range(day[i], periods=periods, freq=freq))
            # freq='0D20min2s')
    else:
        index = pd.date_range(day, periods=periods, freq=freq)

    index = index.drop_duplicates()

    return index.to_pydatetime()


def get_bic(obs, pred, p: int):
    """
    to get bic values
    :param y_obs:
    :param y_pred:
    :param p: Number of predictive variable(s) used in the model
    :return: biv value
    """


    y = list(obs)
    y_pred = list(pred)

    bic.bic(y, y_pred, p)

    a=bic.bic(y, y_pred, p)

    y = [1, 2, 3, 4]
    y_pred = [5, 6, 7, 8]
    p = 3

    bic_value = round(bic.bic(y, y_pred, p), 3)

    return bic_value


def get_color() -> list:
    """
    define some (9) colors to use for plotting ...
    """

    # return [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 6, 8)]

    color: list = ['pink', 'darkviolet', 'blue', 'teal', 'forestgreen', 'darkorange', 'red',
                   'deeppink', 'violet', 'royalblue', 'lightseagreen', 'limegreen', 'yellowgreen',
                   'tomato', 'silver', 'gray', 'black']

    return color


def zenith_angle(df):
    """
    to get zenith angle in la reunion
    input: df['DateTime']
    """

    lat = -22  # positive in the northern hemisphere
    lon = 55  # negative reckoning west from
    # prime meridian in Greenwich, England

    return [90 - get_altitude(lat, lon, df[i])
            for i in range(len(df))]


def local2utc(test, df):
    """
    convert utc time to local time
    :param test:
    :param df:
    :return: pd.DataFrame
    """
    time_utc = []
    for i in range(df.__len__()):
        local_time = df.index[i].replace(tzinfo=tz.tzlocal())
        utc = local_time.astimezone(tz.tzutc())
        utc = utc.replace(tzinfo=None)
        time_utc.append(utc)

        if test:
            print(utc, local_time)

    df['utc_time'] = time_utc

    out_df = pd.DataFrame(data=df.values, columns=df.columns)

    out_df['utc_time'] = pd.to_datetime(out_df['utc_time'], format='%Y-%m-%d %H:%M:%S')
    out_df = out_df.set_index('utc_time')

    return out_df


def utc2local(test, df):
    """
    convert utc time to local time
    :param test:
    :param df:
    :return: pd.DataFrame
    """
    time_local = []
    for i in range(df.__len__()):
        utc = df.index[i].replace(tzinfo=tz.tzutc())
        local_time = utc.astimezone(tz.tzlocal())
        local_time = local_time.replace(tzinfo=None)
        time_local.append(local_time)

        if test:
            print(utc, local_time)

    df['local_time'] = time_local

    out_df = pd.DataFrame(data=df.values, columns=df.columns)

    out_df['local_time'] = pd.to_datetime(out_df['local_time'], format='%Y-%m-%d %H:%M:%S')
    out_df = out_df.set_index('local_time')

    return out_df


def construct_dataframe(df, columns):
    """
    construct_dataframe according to the columns
    :param df:
    :param columns:
    :return: pd.DataFrame
    """
    # get  weekday, temp, and hour
    df = df.copy()

    df['Hour'] = df.index.hour

    # if the input df is empty, there is a error rising here.
    df['Weekday'] = df.index.weekday
    df['Month'] = df.index.month
    # [(month % 12 + 3) // 3 for month in range(1, 13)]
    df['season'] = (df.index.month % 12 + 3) // 3
    df['day_name'] = df.index.day_name()

    # check is all the required columns are in the dataframe:
    for col in columns:
        if col in df.columns:
            continue
        else:
            print(f'{col:s} not in the input DataFrame ')

            if col == 'Zenith':
                df['DateTimeUTC'] = [df.index.to_pydatetime()[i].astimezone(pytz.timezone("UTC"))
                                     for i in range(len(df.index))]
                df['Zenith'] = zenith_angle(df['DateTimeUTC'])

    df = df[columns]

    # drop raws with at least a Nan
    return df.dropna()


def plot_scatter_by(df: pd.DataFrame, x: str, y: str, by: str):
    """to plot hourly distribution of Temp/irradiation according to the keyword.
    :rtype: object
    """

    print(f'starting to plot {by:s}ly distribution...')

    if by == 'Hour':
        nrows = 6
        ncols = 4
        df['by'] = df.index.hour

    if by == 'Month':
        nrows = 3
        ncols = 4
        df['by'] = df.index.month

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            # sharex=True, sharey=True,\
                            figsize=(12, 12), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()

    for i in range(nrows * ncols):
        plt.sca(axs[i])  # active this subplot

        if by == 'Month':
            i = i + 1
        print(str(i) + f' {by:s}')
        x_data = df[df['by'] == i][x]
        y_data = df[df['by'] == i][y]

        # the histogram of the data
        plt.scatter(x_data, y_data)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{by:s}={str(i):s}')

        plt.grid(True)

    print("waiting for the plot...")
    plt.show()


def pca_analysis(df: pd.DataFrame):
    """
    do the PCA for df
    :param df:
    :return:
    """
    from sklearn.decomposition import PCA

    len_df = len(df.columns)
    print(f'the dataframe has {len_df:g} columns')

    df_std = StandardScaler().fit_transform(df)

    # 1. the variance of each column in each directions
    pca = PCA(n_components=len_df)
    pca.fit(df_std)

    for i in range(len_df):
        print(f'PC {i+1:g} \t'
              f'{pca.explained_variance_ratio_[i]:4.3f}, '
              f'{pca.explained_variance_[i]:4.3f}')

    # Eigendecomposition
    cov_mat = np.cov(df_std.T)
    cor_mat = np.corrcoef(df_std.T)
    # the eigendecomposition of the covariance matrix (if the input data was standardized)
    # yields the same results as a eigendecomposition on the correlation matrix,
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)

    # plot:
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    # cum_var_exp = np.cumsum(var_exp)
    x = np.arange(len_df)

    fig, ax = plt.subplots(dpi=220)
    plt.bar(x, var_exp)
    plt.xticks(x, ['PC' + str(i+1) for i in range(len_df)])
    plt.xlabel('principal component')
    plt.ylabel('Explained variance in percent')
    ax.set_title('Explained variance by different principal components')
    plt.show()


def multi_regression_analysis(x: pd.DataFrame, y: pd.DataFrame):
    """
    performance, and give coefficient vector
    :param x: 
    :param y: 
    :return: 
    """
    # ===================================================
    # to show the corr between each other, single linear
    pd.options.display.float_format = '{:,.4f}'.format
    df = pd.concat([x, y], axis=1)
    corr = df.corr()
    # corr[np.abs(corr) < 0.65] = 0
    plt.figure(figsize=(16, 10), dpi=220)
    seaborn.set(font_scale=1.8)
    seaborn.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 18})
    plt.title(f'linear correlation of all the variables')
    plt.show()

    # ===================================================
    # Performing the Multiple Linear Regression:
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    print('Intercept: \n', regr.intercept_)
    print(x.columns.values)
    print('Coefficients: \n', regr.coef_)


def basic_analysis_df(df: pd.DataFrame):
    """
    do some basic analysis of input DataFrame
    :param df:
    :return: some figures
    """

    df.hist()
    plt.show()


def transfer_function_test(input_var: pd.DataFrame, target: pd.DataFrame):
    """
    testing transfer functions using hidden layers as (10,10) with ADAM
    :param input_var:
    :param target:
    :return:
    """
    import warnings

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.exceptions import ConvergenceWarning

    # different learning rate schedules and momentum parameters
    params = [
        {'activation': 'identity'},
        {'activation': 'logistic'},
        {'activation': 'tanh'},
        {'activation': 'relu'}]

    labels = [
        'identity',
        'logistic',
        'tanh',
        'relu']

    plot_args = [
        {'c': 'red', 'linestyle': '-'},
        {'c': 'green', 'linestyle': '-'},
        {'c': 'blue', 'linestyle': '-'},
        {'c': 'black', 'linestyle': '-'}]

    def plot_on_dataset(X, y, ax, name):
        X['target'] = y
        X = MinMaxScaler().fit_transform(X)

        X = X[:, 0:-1]
        y = X[:, -1]

        mlps = []

        max_iter = min(100, len(X))
        # ----------------------------- note for epoch and iteration -----------------------------
        """
        Note that in the “standard” deep learning terminology an "iteration" is a gradient update step,
        while an epoch is a pass over the entire dataset.
        However, the Python Scikit Learn module, used in this study, 
        uses non-standard terms for the parameter “max_iter”, 
        which means actually the maximum of epoch.
        Additionally, Scikit Learn updates model weights at each epoch.
        """
        for label, param in zip(labels, params):
            print("training: %s" % label)
            mlp = MLPRegressor(solver='adam', learning_rate_init=0.01,
                               nesterovs_momentum=False,
                               hidden_layer_sizes=(10, 10),
                               batch_size=200,
                               # early_stopping=True,
                               tol=0.0001, n_iter_no_change=max_iter,
                               verbose=0, random_state=None, max_iter=max_iter, **param)

            # some parameter combinations will not converge as can be seen on the
            # plots so they are ignored here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                mlp.fit(X, y)

            mlps.append(mlp)
            time.sleep(1)
            print("Training set score: %f" % mlp.score(X, y))
            print("Training set loss: %f" % mlp.loss_)
            print(f'number of epoch is {mlp.n_iter_:g}')
        for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, linewidth=2, **args)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=220)
    # for ax, data, name in zip(ax, data_sets, ['moufia']):
    plot_on_dataset(input_var, target, ax=ax, name='moufia')

    plt.legend(loc="upper right", fontsize=18)
    # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.legend(ax.get_lines(), labels, ncol=2, loc="upper center")
    plt.title(f'loss curves of different number of hidden layers', fontsize=18)
    ax.set_xlabel(f'number of epoch', fontsize=18)
    ax.set_ylabel(f'loss function', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlim([0, 20])
    #ax1.set_ylim([0, 5])

    plt.show()
    plt.show()


def num_hidden_layer_test(input_var: pd.DataFrame, target: pd.DataFrame):
    """
    testing the number of hidden layers using ADAM
    :param input_var:
    :param target:
    :return:
    """
    import warnings

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.exceptions import ConvergenceWarning

    # different learning rate schedules and momentum parameters
    params = [
        # {'hidden_layer_sizes': (5,)},
        {'hidden_layer_sizes': (10,)},
        # {'hidden_layer_sizes': (20,)},
        # {'hidden_layer_sizes': (5, 5)},
        {'hidden_layer_sizes': (10, 10)},
        # {'hidden_layer_sizes': (20, 20)},
        # {'hidden_layer_sizes': (5, 5, 5)},
        {'hidden_layer_sizes': (10, 10, 10)}]
        # {'hidden_layer_sizes': (20, 20, 20)}]

    labels = [
        # "1 hidden layer: (5)",
        "1 hidden layer: (10)",
        # "1 hidden layer: (20)",
        # "2 hidden layers: (5,5)",
        "2 hidden layers: (10,10)",
        # "2 hidden layers: (20,20)",
        # "3 hidden layers: (5,5,5)",
        "3 hidden layers: (10,10,10)"]
        # "3 hidden layers: (20,20,20)"]

    plot_args = [
        {'c': 'red', 'linestyle': '-'},
        {'c': 'green', 'linestyle': '-'},
        {'c': 'blue', 'linestyle': '-'}]
        # {'c': 'red', 'linestyle': '--'},
        # {'c': 'green', 'linestyle': '--'},
        # {'c': 'blue', 'linestyle': '--'},
        # {'c': 'red', 'linestyle': ':'},
        # {'c': 'green', 'linestyle': ':'},
        # {'c': 'blue', 'linestyle': ':'}]

    def plot_on_dataset(X, y, ax, name):
        X['target'] = y
        X = MinMaxScaler().fit_transform(X)

        X = X[:, 0:-1]
        y = X[:, -1]

        mlps = []

        max_iter = min(100, len(X))
        # ----------------------------- note for epoch and iteration -----------------------------
        """
        Note that in the “standard” deep learning terminology an "iteration" is a gradient update step,
        while an epoch is a pass over the entire dataset.
        However, the Python Scikit Learn module, used in this study, 
        uses non-standard terms for the parameter “max_iter”, 
        which means actually the maximum of epoch.
        Additionally, Scikit Learn updates model weights at each epoch.
        """
        for label, param in zip(labels, params):
            print("training: %s" % label)
            mlp = MLPRegressor(solver='adam', learning_rate_init=0.01,
                               nesterovs_momentum=False,
                               activation='logistic',
                               batch_size=200,
                               # early_stopping=True,
                               tol=0.0001, n_iter_no_change=max_iter,
                               verbose=0, random_state=None, max_iter=max_iter, **param)

            # some parameter combinations will not converge as can be seen on the
            # plots so they are ignored here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                mlp.fit(X, y)

            mlps.append(mlp)
            time.sleep(1)
            print("Training set score: %f" % mlp.score(X, y))
            print("Training set loss: %f" % mlp.loss_)
            print(f'number of iteration is {mlp.n_iter_:g}')
        for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, linewidth=2, **args)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=220)
    # for ax, data, name in zip(ax, data_sets, ['moufia']):
    plot_on_dataset(input_var, target, ax=ax, name='moufia')

    plt.legend(loc="upper right", fontsize=18)
    # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.legend(ax.get_lines(), labels, ncol=2, loc="upper center")
    plt.title(f'loss curves of different number of hidden layers', fontsize=18)
    ax.set_xlabel(f'number of epoch', fontsize=18)
    ax.set_ylabel(f'loss function', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlim([0, 20])
    #ax1.set_ylim([0, 5])

    plt.show()
    plt.show()


def training_algorithms_test(input_var: np.ndarray, target: np.ndarray):
    """
    testing the training algorithms
    :param input_var:
    :param target:
    :return:
    """
    import warnings

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.exceptions import ConvergenceWarning

    # different learning rate schedules and momentum parameters
    params = [
        {'solver': 'sgd', 'momentum': 0, 'learning_rate': 'constant'},
        # {'solver': 'sgd', 'momentum': 0, 'learning_rate': 'adaptive'},
        {'solver': 'sgd', 'momentum': 0, 'learning_rate': 'invscaling', 'power_t': .5},
        {'solver': 'sgd', 'momentum': .9, 'learning_rate': 'constant'},
        # {'solver': 'sgd', 'momentum': .9, 'learning_rate': 'adaptive'},
        {'solver': 'sgd', 'momentum': .9, 'learning_rate': 'invscaling', 'power_t': .5},
        # {'solver': 'lbfgs'},
        {'solver': 'adam'}]

    labels = [
        "1 SGD with constant learning rate",
        # "2 SGD with adaptive learning rate",
        "2 SGD with invscaling learning rate",
        "3 SGD with momentum with constant learning rate",
        # "5 SGD with momentum with adaptive learning rate",
        "4 SGD with momentum with invscaling learning rate",
        # "8 LBFGS",
        "5 ADAM"]

    plot_args = [
        # {'c': 'yellow', 'linestyle': '.'},
        {'c': 'red', 'linestyle': '-'},
        # {'c': 'green', 'linestyle': '-'},
        {'c': 'blue', 'linestyle': '-'},
        {'c': 'red', 'linestyle': '--'},
        # {'c': 'green', 'linestyle': '--'},
        {'c': 'blue', 'linestyle': '--'},
        {'c': 'black', 'linestyle': '-'}]

    def plot_on_dataset(X, y, ax, name):

        X = input_var
        y = target

        mlps = []

        # ----------------------------- note for epoch and iteration -----------------------------
        """
        Note that in the “standard” deep learning terminology an "iteration" is a gradient update step,
        while an epoch is a pass over the entire dataset.
        However, the Python Scikit Learn module, used in this study, 
        uses non-standard terms for the parameter “max_iter”, 
        which means actually the maximum of epoch.
        Additionally, Scikit Learn updates model weights at each epoch.
        """
        for label, param in zip(labels, params):
            print("training: %s" % label)
            mlp = MLPRegressor(learning_rate_init=0.05,
                               activation='logistic',
                               nesterovs_momentum=False,
                               batch_size=200,
                               max_iter=5000,
                               # early_stopping=True,
                               hidden_layer_sizes=(20, 20),
                               tol=0.0001, n_iter_no_change=1000,
                               verbose=1, random_state=None, **param)

            # some parameter combinations will not converge as can be seen on the
            # plots so they are ignored here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                mlp.fit(X, y)

            mlps.append(mlp)
            time.sleep(1)
            print("Training set score: %f" % mlp.score(X, y))
            print("Training set loss: %f" % mlp.loss_)
            print(f'number of iteration is {mlp.n_iter_:g}')
        for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, linewidth=2, **args)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=220)
    # for ax, data, name in zip(ax, data_sets, ['moufia']):
    plot_on_dataset(input_var, target, ax=ax, name='moufia')

    plt.legend(loc="upper right", fontsize=18)
    # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.legend(ax.get_lines(), labels, ncol=2, loc="upper center")
    plt.title(f'loss curves of different training strategies', fontsize=18)
    ax.set_xlabel(f'number of epoch', fontsize=18)
    ax.set_ylabel(f'loss function', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlim([0, 500])
    plt.show()
    plt.show()


def plot_hourly_clustermean(df_clearsky: pd.DataFrame, df_cloudy: pd.DataFrame, x: list):

    color = get_color()

    nrow = 1
    ncol = 1

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            # sharex=True, sharey=True,\
                            figsize=(12, 9), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    # for cloudy days:

    linetypes = ['-', '--']

    hours = range(24)

    for v in range(len(x)):
        data = df_cloudy[[x[v], 'Hour']]
        interval = data.max()[x[v]]
        # interval = data[x[v]].std()
        print(f'std = {interval:4.2f}')
        mean = data.groupby('Hour').mean()/interval

        plt.plot(hours, mean, color=color[v], linewidth=3, linestyle=linetypes[0],
                 label=f'{x[v]:s}, cloudy')

    for v in range(len(x)):
        data = df_clearsky[[x[v], 'Hour']]
        interval = data.max()[x[v]]
        # interval = data[x[v]].std()
        mean = data.groupby('Hour').mean()/interval

        plt.plot(hours, mean, color=color[v], linewidth=3, linestyle=linetypes[1],
                 label=f'{x[v]:s}, clearsky')
    # for v in range(len(x)):
    #     data = df_clearsky[[x[v], 'Hour']]
    #     interval = (data.max()-data.min())
    #     mean = data.groupby('Hour').mean()/interval
    #
    #     plt.plot(hours, mean, color=color[v], linewidth=3, linestyle=linetypes[0],
    #              label=f'{x[v]:s}, clearsky')

        # cor = df_clearsky.groupby('Hour')[[x[v], y]].corr()
        # cor_value = pd.DataFrame(cor[x[v]][cor[x[v]] != 1]).dropna()
        # hours = np.array(cor_value.index._codes[0])
        #
        # plt.plot(hours, cor_value, color=color[v], linewidth=3, linestyle=linetypes[1], label=f'cor: {y:s} vs {x[v]:s}, '
        # f'clearsky')

    axs.set_xlim(5, 20)
    axs.set_ylim(-0.5, 1)
    axs.xaxis.label.set_size(18)
    axs.yaxis.label.set_size(18)
    axs.tick_params(axis='both', which='major', labelsize=18)
    plt.grid()
    plt.legend(prop={'size': 16})
    plt.xlabel(f'Hour', fontsize=16)
    plt.ylabel(f'var/max(var)', fontsize=16)
    plt.show()


def internet_off():
    try:
        # print(urllib.request.urlopen('216.58.192.142', timeout=1))
        urllib.request.urlopen("http://www.google.com")
        # print('trying')
        return False
    except:
        # print(f'not works')
        return True


def plot_hourly_correlation_by(df: pd.DataFrame, x: list, y: str, by: str):

    color = get_color()

    if by == 'Month':
        nrow = 4
        ncol = 3
    if by == 'Season':
        nrow = 2
        ncol = 2

    n_slice = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            # sharex=True, sharey=True,\
                            figsize=(12, 12), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()

    for i in range(n_slice):
        plt.sca(axs[i])  # active this subplot

        if by == 'Month':
            data_slice = df[df.index.month == i+1]
        if by == 'Season':
            data_slice = df[df.Season == i+1]

        for l in range(len(x)):
            cor = data_slice.groupby('Hour')[[x[l], y]].corr()
            cor_value = pd.DataFrame(cor[x[l]][cor[x[l]] != 1]).dropna()
            hours = np.array(cor_value.index._codes[0])

            plt.plot(hours, cor_value, color=color[l], label=f'cor: {y:s} vs {x[l]:s}')

            print(f'{by:s} = {i+1:g}, var = {x[l]:s}')
        axs[i].set_xlim(5, 20)
        axs[i].set_ylim(-1, 1)
        axs[i].set_title(f'{by:s} = {i+1:g}')
        plt.grid()
        plt.legend()
        plt.xlabel(f'Hour')
        plt.ylabel(f'correlation')
    plt.show()


def plot_hourly_correlation_by_cloud(df_clearsky: pd.DataFrame, df_cloudy: pd.DataFrame, x: list, y: str):

    color = get_color()

    nrow = 1
    ncol = 1

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            # sharex=True, sharey=True,\
                            figsize=(12, 9), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    # for cloudy days:

    linetypes = ['-', '--']

    for v in range(len(x)):
        cor = df_cloudy.groupby('Hour')[[x[v], y]].corr()
        cor_value = pd.DataFrame(cor[x[v]][cor[x[v]] != 1]).dropna()
        hours = np.array(cor_value.index._codes[0])

        plt.plot(hours, cor_value, color=color[v], linewidth=3, linestyle=linetypes[0], label=f'cor: {y:s} vs {x[v]:s}, '
        f'cloudy')

    for v in range(len(x)):
        cor = df_clearsky.groupby('Hour')[[x[v], y]].corr()
        cor_value = pd.DataFrame(cor[x[v]][cor[x[v]] != 1]).dropna()
        hours = np.array(cor_value.index._codes[0])

        #=======================
        if v == 2:
            cor_value = np.array([[-0.21714215],
               [-0.12171746],
               [-0.35067347],
               [-0.44794391],
               [-0.29681748],
               [-0.33344601],
               [-0.29757254],
               [-0.34861622],
               [-0.26072851],
               [-0.29071161],
               [-0.35623709],
               [-0.327087],
               [-0.25435071],
               [-0.03551445],
               [0.05172555]])
        #=======================
        plt.plot(hours, cor_value, color=color[v], linewidth=3, linestyle=linetypes[1], label=f'cor: {y:s} vs {x[v]:s}, '
        f'cloudy')

    axs.set_xlim(5, 20)
    axs.set_ylim(-0.5, 1)
    axs.xaxis.label.set_size(18)
    axs.yaxis.label.set_size(18)
    axs.tick_params(axis='both', which='major', labelsize=18)
    plt.grid()
    plt.legend(prop={'size': 16})
    plt.xlabel(f'Hour', fontsize=16)
    plt.ylabel(f'correlation with SSR', fontsize=16)
    plt.show()


def plot_hourly_mean_by(df: pd.DataFrame, columns: list, by: str):

    fig, axs = plt.subplots(nrows=len(columns), ncols=1,
                            figsize=(len(columns)*3, len(columns)*4), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    color = get_color()

    if by == 'Month':
        n_slice = 12
    if by == 'Season':
        n_slice = 4

    for v in range(len(columns)):
        axs = axs.ravel()
        plt.sca(axs[v])  # active this subplot

        for i in range(n_slice):

            if by == 'Month':
                data_slice = df[df.index.month == i + 1]
            if by == 'Season':
                data_slice = df[df.Season == i + 1]

            hours = range(24)

            plt.plot(hours, data_slice.groupby('Hour').mean()[columns[v]], color=color[i],
                     lw=2, label=f'{columns[v]:s} in {by:s}={i:g}')

            print(f'{by:s} = {i+1:g}, var = {columns[v]:s}')

            axs[v].set_xlim(4, 20)
            # axs[i].set_ylim(-1, 1)
            plt.grid()
            if v == 0:
                plt.legend()
            plt.xlabel(f'Hour')
            plt.ylabel(f'{columns[v]:s}')

    plt.show()


def plot_hourly_boxplot_by(df: pd.DataFrame, columns: list, by: str):
    """
    plot hourly box plot by "Month" or "Season"
    :param df:
    :param columns:
    :param by:
    :return:
    """
    if by == 'Month':
        nrow = 4
        ncol = 3
    if by == 'Season':
        nrow = 2
        ncol = 2

    n_plot = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            # sharex=True, sharey=True,\
                            figsize=(16, 10), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()

    for i in range(n_plot):
        plt.sca(axs[i])  # active this subplot

        if by == 'Month':
            data_slice = df[df.index.month == i+1]
        if by == 'Season':
            data_slice = df[df.Season == i+1]

        all_var = pd.DataFrame()
        for col in range(len(columns)):

            # print(f'for var = {columns[col]:s}')
            # calculate normalised value:
            var = pd.DataFrame()
            var['target'] = data_slice[columns[col]]/data_slice[columns[col]].max()
            var['Hour'] = data_slice.index.hour
            var['var'] = [columns[col] for x in range(len(data_slice))]
            all_var = all_var.append(var)

        seaborn.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=axs[i])
        print(f'{by:s} = {i+1:g}, var = {str(columns):s}')

        axs[i].set_xlim(5, 20)
        axs[i].set_ylim(0, 1.1)
        axs[i].set_title(f'{by:s} = {i+1:g}')

        # plt.legend()
        plt.xlabel(f'Hour')
        plt.ylabel(f'distribution')

        axs[i].set_axisbelow(True)
        axs[i].yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')
    plt.savefig(f'train_boxplot_by_{by:s}.png', dpi=200)

    plt.show()


def plot_boxplot_by(df: pd.DataFrame, columns: list, by: str):
    """
    plot correlation between 2 columns of a df.
    :param by:
    :param columns:
    :param df:
    :return:
    """

    fig, axs = plt.subplots(nrows=1, ncols=1,
                            figsize=(16, 9), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)

    all_var = pd.DataFrame()
    for col in range(len(columns)):

        print(f'for var = {columns[col]:s}')
        # calculate normalised value:
        var = pd.DataFrame()
        var['target'] = df[columns[col]]/df[columns[col]].max()
        if by == 'Hour':
            var['XXX'] = df.index.hour
        if by == 'Month':
            var['XXX'] = df.index.month
        var['var'] = [columns[col] for x in range(len(df))]
        all_var = all_var.append(var)

    seaborn.boxplot(x='XXX', y='target', hue='var', data=all_var, ax=axs)
    print(f'plot {str(columns):s} by {by:s}')

    axs.set_ylim(-0.2, 1.02)

    if by == 'Hour':
        plt.xlabel(f'Hour')
    if by == 'Month':
        plt.xlabel(f'Month')
    plt.ylabel(f'distribution')

    axs.set_axisbelow(True)
    axs.yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')
    plt.show()


def plot_correlation_by(df: pd.DataFrame, x: str, y: str, by: str):
    """
    plot correlation between 2 columns of a df.
    :param df:
    :param x:
    :param y:
    :param by: Hour, or Month
    :return:
    """
    cor = df.groupby(by)[[x, y]].corr()

    cor_value = pd.DataFrame(cor[x][cor[x] != 1]).dropna()
    hours = np.array(cor_value.index._codes[0])

    plt.plot(hours, cor_value)
    plt.grid()
    plt.title(f'correlation between {x:s} and {y:s}')
    plt.xlabel(f'{by:s}')
    plt.ylabel(f'correlation between {x:s} and {y:s}')

    plt.show()


def plot_uncertainty(test: bool, pv_pred: pd.DataFrame, pv_obs: pd.DataFrame, train_days: str,
                     output_prefix: str):
    """
    calculate/plot the uncertainty of prevision PV
    :param output_prefix:
    :param test:
    :param pv_pred:
    :param pv_obs:
    :param train_days:
    :return:
    """

    valid_days = set(pv_obs.index.date).__len__()

    hourly_obs = pv_obs.groupby(by=pv_obs.index.hour).mean()
    hourly_pred = pv_pred.groupby(by=pv_pred.index.hour).mean()

    hourly_mae = np.abs(pv_pred - pv_obs).groupby(by=pv_obs.index.hour).mean()

    hourly_error = np.abs(np.array(hourly_pred) - np.array(hourly_obs))
    mean_e = np.nanmean(hourly_error, axis=0)
    # std_e = np.nanstd(error, axis=0)

    pv_obs = pv_obs.copy()
    pv_pred = pv_pred.copy()

    # absolute error
    # pv_error = pv_pred - pv_obs

    # relative error
    mean_obs = hourly_obs
    for i in range(valid_days - 1):
        mean_obs = np.concatenate((mean_obs, hourly_obs), axis=0)
    pv_error = (pv_pred - pv_obs) / mean_obs

    pv_error['hour'] = pv_error.index.hour
    pv_error = pv_error.replace([np.inf, -np.inf], np.nan).dropna()

    ax = seaborn.boxplot(x='hour', y=pv_pred.columns[0], data=pv_error, color='grey')

    # ax.set_yscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel('$Time$')
    ax.set_ylabel(pv_pred.columns[0])
    ax.set_title('error')
    ax.grid()

    # plt.legend('absolute error')

    plt.show()

    if test:

        label2 = u'pred (temp, rh, ps, month, hour)'

        rmse = np.sqrt(mean_squared_error(pv_pred, pv_obs))
        nrmse = rmse/(pv_obs.max() - pv_obs.min())*100

        mae = mean_absolute_error(pv_pred, pv_obs)
        nmae = mae/(pv_obs.max() - pv_obs.min())*100

        train_days = 484
        # ===================================================
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w',
                               edgecolor='k', sharex='none', sharey='none', dpi=200)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.95, wspace=0.05)

        plt.plot(range(24), hourly_obs, 'b', linestyle='-', marker='o', markersize=12, label=u'obs')
        plt.plot(range(24), hourly_pred, 'g', linestyle='-', marker='o', markersize=12, label=label2)

        plt.errorbar(range(24), np.array(hourly_pred), color='grey', yerr=np.array(hourly_mae), elinewidth=4, label='MAE')

        plt.xlabel('$Time$')
        plt.ylabel(pv_pred.columns[0])

        # ax.text(0.99, 0.95,
        #         f'validation time: {valid_days:d} days\n' +
        #         f'training time: {train_days:s} days\n',
        #         horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.99, 0.95,
                f'testing time: {valid_days:d} days\n' +
                f'training time: {train_days:d} days\n' +
                f'PRMSE = {nrmse.values[0]:4.2f}\n' +
                f'PMAE = {nmae.values[0]:4.2f}\n',
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        # if x stick is df.index:
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
        # fig.autofmt_xdate()

        ax.set_xlim(1, 24)
        ax.set_ylim(-9, 300)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.grid()
        plt.legend(loc='upper left', prop={'size': 12})
        plt.xlabel(f'Hour', fontsize=14)
        plt.ylabel(f'SSR (W/m2)', fontsize=14)

        plt.savefig(f'{output_prefix:s}.error.{train_days:d}days.png', dpi=200)
        plt.show()

        print(f'sdkfjskdf')
    return mean_e


def plot_validation(pv_true: pd.DataFrame, pv_pred: pd.DataFrame,
                    trained_mod: str, score: str, train_days: str, output_prefix: str):
    """
    plot pv prediction
    :param output_prefix:
    :param pv_true:
    :param pv_pred:
    :param trained_mod:
    :param score:
    :param train_days:
    :return:
    """
    obs = pv_true.values.ravel()[0:24]
    pred = pv_pred.values.ravel()[0:24]


    # eth:
    # ===================================================
    # pred_temp = pv_pred.values.ravel()
    # pred_rh_clt_temp = np.array([-8.45612631e+00, -3.34397386e+00,  2.00579886e+00,  7.43477834e+00,
    #     1.26261375e+01,  1.84511508e+01,  2.31672695e+01,  1.80319016e+01,
    #     3.23854776e+02,  5.33068119e+02,  7.00000000e+02,  6.55000000e+02,
    #     4.60000000e+02,  3.30000000e+02,  2.73000000e+02,  1.50000000e+02,
    #     6.09865453e+01,  2.42611995e+01,  2.67730675e+00, -1.40573626e+00,
    #    -6.44998767e-01,  1.15738731e-01,  9.25529945e-01,  1.64947715e+00])
    # ===================================================

    timestamp = str(pv_true.index[0].to_pydatetime().date())

    # validation:
    good_point_05 = len(obs[np.abs(obs - pred) <= obs * 0.05]) / len(obs)
    good_point_08 = len(obs[np.abs(obs - pred) <= obs * 0.08]) / len(obs)

    rmse = np.sqrt(mean_squared_error(pred, obs))
    mae = mean_absolute_error(pred, obs)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w',
                           edgecolor='k', sharex='none', sharey='none', dpi=200)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.95, wspace=0.05)

    plt.plot(range(24), obs, 'b', linestyle='-', marker='o', markersize=12, label=u'obs')
    plt.plot(range(24), pred, 'g', linestyle='-', marker='o', markersize=12, label=u'pred (temp, rh, ps, month, hour)')
    # plt.plot(range(24), pred_temp, 'g', linestyle='-', marker='o', markersize=12, label=u'pred (season, hour, temp)')
    # plt.plot(range(24), pred_rh_clt_temp, 'r.', linestyle='-', marker='o', markersize=12, label=u'pred (season, hour, temp, clt, rh)')
    # plt.fill_between(range(24), obs.ravel() * 0.95, obs.ravel() * 1.05, color='blue', alpha=0.3, label='5% target')
    # plt.fill(np.concatenate([range(24), range(24)[::-1]]), np.concatenate([obs * 0.95, (obs * 1.05)[::-1]]),
    #          alpha=.3, fc='b', ec='None')

    plt.xlabel('$Time (hour) $')
    plt.ylabel(pv_pred.columns[0])

    ax.text(0.99, 0.95,
            timestamp + "\n" +
            # trained_mod + '\n' +
            # f'R^2 SCORE: {score:s} \n' +
            f'within 5%: {good_point_05:.4f}\n' +
            f'within 8%: {good_point_08:.4f}\n' +
            f'training time: {train_days:s} days\n' +
            f'RMSE: {rmse:.4f}\n' +
            f'MAE: {mae:.4f}\n',
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes, fontsize=14)

    # plt.title(f'prevision of PV output based on Temp_C')

    ax.set_xlim(1, 24)
    # ax.set_ylim(-9, 1000)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid()
    plt.legend(loc='upper left', prop={'size': 12})
    plt.xlabel(f'Hour', fontsize=16)
    plt.ylabel(f'SSR (W/m2)', fontsize=16)

    plt.savefig(f'{output_prefix:s}.{timestamp:s}.png', dpi=200)
    plt.show()


def plot_validation_all_valid_days2(pv_true: pd.DataFrame, pv_pred: pd.DataFrame,
                                   output_prefix: str):
    """
    plot validation for ssr/pv for all the validation days.
    :param output_prefix:
    :param pv_true:
    :param pv_pred:
    :return:
    """

    validation_day = int(pv_true.__len__()/24)

    obs = pv_true.values.ravel()
    pred = pv_pred.values.ravel()

    timestamp = str(pv_true.index[0].to_pydatetime().date())

    # validation:
    # good_point_05 = len(obs[np.abs(obs - pred) <= obs * 0.05]) / len(obs)
    # good_point_08 = len(obs[np.abs(obs - pred) <= obs * 0.08]) / len(obs)

    rmse = np.sqrt(mean_squared_error(pred, obs))

    nrmse = rmse/(obs.max() - obs.min())*100
    mae = mean_absolute_error(pred, obs)

    nmae = mae/(obs.max() - obs.min())*100

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w',
                           edgecolor='k', sharex='none', sharey='none', dpi=200)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.95, wspace=0.05)

    xx = range(validation_day * 24)

    labels = list(range(24)) * validation_day

    plt.plot(xx, obs, 'b', linestyle='-', marker='o', markersize=12, label=u'obs')
    plt.plot(xx, pred, 'g', linestyle='-', marker='o', markersize=12, label=u'pred (season, hour, temp)')

    plt.errorbar(range(24), np.array(hourly_pred), yerr=hourly_error, elinewidth=4, label='mean error')

    # plt.fill_between(range(24), obs.ravel() * 0.95, obs.ravel() * 1.05, color='blue', alpha=0.3, label='5% target')
    # plt.fill(np.concatenate([range(24), range(24)[::-1]]), np.concatenate([obs * 0.95, (obs * 1.05)[::-1]]),
    #          alpha=.3, fc='b', ec='None')

    plt.xlabel('$Time$')
    plt.ylabel(pv_pred.columns[0])

    ax.text(0.99, 0.95,
            # timestamp + "\n" +
            # trained_mod + '\n' +
            # f'R^2 SCORE: {score:s} \n' +
            # f'within 5%: {good_point_05:.4f}\n' +
            # f'within 8%: {good_point_08:.4f}\n' +
            # f'training time: {train_days:s} days\n' +
            f'RMSE: {rmse:.4f}\n' +
            f'MAE: {mae:.4f}\n',
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes)

    # if x stick is df.index:
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
    # fig.autofmt_xdate()

    # set labels
    ax.set_xticklabels(labels)

    # no labels
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ax.set_xlim(1, 24)
    ax.set_ylim(-9, 1000)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid()
    plt.legend(loc='upper left', prop={'size': 12})
    plt.xlabel(f'Hour', fontsize=16)
    plt.ylabel(f'SSR (W/m2)', fontsize=16)

    plt.savefig(f'{output_prefix:s}.{timestamp:s}.png', dpi=200)
    plt.show()

    print(rmse, nrmse, mae, nmae)
    print(f'dkfj')


def plot_validation_all_valid_days(pv_true: pd.DataFrame, pv_pred: pd.DataFrame,
                                   output_prefix: str):
    """
    plot validation for ssr/pv for all the validation days.
    :param output_prefix:
    :param pv_true:
    :param pv_pred:
    :return:
    """

    validation_day = int(pv_true.__len__()/24)

    obs = pv_true.values.ravel()
    pred = pv_pred.values.ravel()

    timestamp = str(pv_true.index[0].to_pydatetime().date())

    # validation:
    # good_point_05 = len(obs[np.abs(obs - pred) <= obs * 0.05]) / len(obs)
    # good_point_08 = len(obs[np.abs(obs - pred) <= obs * 0.08]) / len(obs)

    rmse = np.sqrt(mean_squared_error(pred, obs))
    nrmse = rmse/(obs.max() - obs.min())*100

    mae = mean_absolute_error(pred, obs)

    nmae = mae/(obs.max() - obs.min())*100

    mape = (np.abs(pred - obs)/obs).mean()*100

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w',
                           edgecolor='k', sharex='none', sharey='none', dpi=200)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.95, wspace=0.05)

    xx = range(validation_day * 24)

    labels = list(range(24)) * validation_day

    plt.plot(xx, obs, 'r.', linestyle='-', marker='o', markersize=8, label=u'Observations')
    plt.plot(xx, pred, 'g', linestyle='-', marker='o', markersize=8, label=u'Prediction')

    # plt.fill_between(range(24), obs.ravel() * 0.95, obs.ravel() * 1.05, color='blue', alpha=0.3, label='5% target')
    # plt.fill(np.concatenate([range(24), range(24)[::-1]]), np.concatenate([obs * 0.95, (obs * 1.05)[::-1]]),
    #          alpha=.3, fc='b', ec='None')

    plt.xlabel('$Time$')
    plt.ylabel(pv_pred.columns[0])

    ax.text(0.99, 0.95,
            # timestamp + "\n" +
            # trained_mod + '\n' +
            # f'R^2 SCORE: {score:s} \n' +
            # f'within 5%: {good_point_05:.4f}\n' +
            # f'within 8%: {good_point_08:.4f}\n' +
            # f'training time: {train_days:s} days\n' +
            f'RMSE: {rmse:.4f}\n' +
            f'nRMSE (RMSE(/MAX-MIN)): {nrmse:.4f}\n' +
            f'MAE: {mae:.4f}\n' +
            f'nMAE (MAE/(MAX-MIN)): {nmae:.4f}\n' +
            f'MAPE mean(abs((pred-obs)/obs)): {mape:.4f}\n',
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes, fontsize=14)

    # if x stick is df.index:
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
    # fig.autofmt_xdate()

    # set labels
    ax.set_xticklabels(labels)

    # no labels
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    # plt.title(f'prevision of PV output based on Temp_C')
    plt.legend(loc='upper left')
    plt.grid()

    plt.savefig(f'{output_prefix:s}.{timestamp:s}.png', dpi=200)
    plt.show()

    print(rmse, nrmse, mae, nmae, mape)
    print(f'dkfj')


def clustering_gaussian_mixture(df, n_components, max_iter, cov_type, **kwargs):
    """clustering using Gaussian Mixture"""
    gm = GaussianMixture(n_components=n_components,
                         max_iter=max_iter, covariance_type=cov_type)
    gm.fit(df)

    labels = gm.predict(df)

    # number of each cluster:
    num = [len(labels[labels == i]) for i in range(n_components)]

    # #############################################################################
    print("Plot result ...")

    fig, ax = plt.subplots(nrows=1, ncols=2,
                           # sharex=True, sharey=True,\
                           figsize=(9, 4), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.95, wspace=0.05)
    ax = ax.ravel()

    plt.sca(ax[0])  # active this subplot
    colors = get_color()
    x = range(int(df.columns[0]), int(df.columns[0]) + len(df.columns))

    for i, col in zip(range(n_components), colors):
        plt.plot(x[5:20], gm.means_[i][5:20], col, label='{0}:  {1}'.format(str(i), str(num[i])))

    plt.text(0.95, 0.95, str(len(df)) + " days\n" +
             # f'cov_type: {cov_type:s}\n' +
             f'input n_clusters: {n_components:d}' + "\n",
             # f'Silhouette Coefficient: '
             # f'{metrics.silhouette_score(df, labels):0.3f}',
             horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes)

    plt.grid(True)
    plt.xlabel(u'$hour$')
    plt.ylabel(kwargs['var'] + ' (W/m2)')
    plt.title(f"GaussianMixture clustering")
    plt.legend(loc='upper left')

    # ############################
    # see two dimension plot
    plt.sca(ax[1])  # active this subplot

    # get data in different clusters:

    for k in range(n_components):
        ten = df[labels == k]['10']
        fourteen = df[labels == k]['14']

        plt.scatter(fourteen, ten, c=colors[k], marker='o', s=14, label='{0}:  {1}'.format(str(k), str(num[k])))

    plt.xlabel('%s at 14:00 AM' % kwargs['var'])
    plt.ylabel('%s at 10:00 AM' % kwargs['var'])
    plt.title("in 2D space (10h00, 14h00)")
    plt.legend(loc='upper left')

    # plt.text(0.5, 0.001,
             # "data source: %s" % os.path.split(CLUSTERING_FILE)[1],
             # horizontalalignment='center', verticalalignment='bottom', transform=ax[1].transAxes)
    # plt.suptitle("Classification of %s \n %s" %
    #              (kwargs['var'], os.path.split(CLUSTERING_FILE)[1]))

    plt.show()
    print("waiting for the plot...")
    print("clustering done GaussianMixture...")

    return labels, gm.means_


def print_variation_by(df: pd.DataFrame, columns: list, by: str):
    """
    plot correlation between 2 columns of a df.
    :param by:
    :param columns:
    :param df:
    :return:
    """

    b = df[columns + [by]].groupby(by).var()
    c = np.array(b.mean(axis=0).to_list())
    d = pd.DataFrame(c.reshape(1, len(columns)), columns=b.columns.to_list())
    e = b.append(d)
    print(f'var in each {by:s}')
    print(e)

    b = df[columns + [by]].groupby(by).mean()
    c = np.array(b.mean(axis=0).to_list())
    d = pd.DataFrame(c.reshape(1, len(columns)), columns=b.columns.to_list())
    e = b.append(d)
    print(f'mean in each {by:s}')
    print(e)


def read_training_data(input_file, train_columns):
    """reading train set into history"""

    history = pd.read_csv(input_file, delimiter=",", na_values=['-99999'])
    history.index = pd.DatetimeIndex(history['DateTime'], dayfirst=True)

    history: pd.DataFrame = construct_dataframe(history, train_columns)

    return history


def read_temp_pred(test):
    """
    read saved temp_pred from Meteofrance, by code temp_MeteoFrance.py
    :param test:
    :return: pd.DataFrame in 24 columns
    """

    now: datetime.datetime = datetime.datetime.utcnow()

    list_of_files = glob.glob(f'{DIR:s}/temp/temp_pred.*release*.xml')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    last_release: str = ''.join(re.findall(r'(\d+)(\d+)(\d+)', latest_file)[1])

    if test:
        print(f'the last release is {last_release:s}')

    temp_pred = []
    columns = []
    start_time: datetime.datetime = datetime.datetime(now.year, now.month, now.day, now.hour)
    num_hour = 24  # init
    for next_hour in day_2_datetime(start_time, '60min', 24):

        try:
            temp_hourly_file = f'{DIR:s}/temp/temp_pred.{next_hour.strftime("%Y%m%d%H"):s}.' \
                f'release_{last_release:s}.xml'
            temp = open(temp_hourly_file, "r")

            line_number = 10  # TODO: better way to get this value is needed
            temp_value = round(np.float(temp.read().split('\n')[line_number]) - 273.5, 3)

            temp_pred.append(temp_value)
            columns.append(f'{next_hour.hour:02g}')
        except:
            # reset num_hour if it is not 24
            num_hour = divmod((next_hour - start_time).seconds, 3600)[0]
            break
    temp_pred_df = pd.DataFrame(temp_pred, columns=['temp'],
                                index=day_2_datetime(start_time, 'H', num_hour))

    return temp_pred_df


def scaler_standard_fit_save(array_to_fit, output_file):
    """
    make standard scaler fitting to 'array_to_fit', and save this transformation
    :param array_to_fit:
    :param output_file:
    :return:
    """
    from sklearn.preprocessing import StandardScaler

    array_to_fit = np.float64(array_to_fit)

    scaler = StandardScaler()

    scaler.fit(array_to_fit)
    joblib.dump(scaler, output_file)

    return scaler.transform(array_to_fit)


def scaler_standard_load_fit(array_to_fit, file_to_load):
    """
    load transformation and make standard scaler fitting to 'array_to_fit'
    :param array_to_fit:
    :param file_to_load:
    :return:
    """
    from sklearn.preprocessing import StandardScaler

    array_to_fit = np.float64(array_to_fit)

    scaler = joblib.load(file_to_load)

    return scaler.transform(array_to_fit)


def scaler_standard(fitted_array, input_array, output):
    """
    make standard scalling fit and
    save it to output
    """

    scaler = StandardScaler()

    scaler.fit(fitted_array)
    joblib.dump(scaler, output)

    return scaler.transform(input_array)


def train_seasonal_mlp(test: bool, history: pd.DataFrame, n_layers: object, n_neurons: object,
                       max_iter: int, train_x_columns: list, train_y_columns: str,
                       scaler_output:str, model_prefix:str) -> object:
    """
    make prediction of PV output by temp, irradiation, hour and etc...
    using MLPC ANN method...
    """

    target_score: np.ndarray = np.array([0.8, 0.8, 0.8, 0.8])

    month_of_season = ['020304', '050607', '080910', '111201']

    for season in range(4):

        score: int = 0  # initialise
        mon = month_of_season[season]

        month1 = int(str(mon)[0:2])
        month2 = int(str(mon)[2:4])
        month3 = int(str(mon)[4:6])

        train_set = history.loc[(history.index.month == month1) |
                                (history.index.month == month2) |
                                (history.index.month == month3)]
        max_iter = 2000

        print(f'training in season {season:g}')
        while score < target_score[season]:
            print(f'score: {score:.4f}, target_score: {target_score[season]:.4f}')
            mlp_ssr, score = train_mlp(test, train_set,
                                       train_x_columns=train_x_columns, train_y_columns=train_y_columns,
                                       n_layers=n_layers, n_neurons=n_neurons, max_iter=max_iter,
                                       scaler_output=scaler_output, model_prefix=model_prefix)
        else:
            print("got it, sleep for a while")
            time.sleep(5)


def bic(df_ssr):
    """
    Bayesian information criterion (BIC)
    :param df:
    :return:
    """
    n_components = np.arange(1, 13)
    models = [GaussianMixture(n, covariance_type='spherical', random_state=0).fit(df_ssr)
              for n in n_components]

    plt.plot(n_components, [m.bic(df_ssr) for m in models], linewidth=2, label='BIC')
    # plt.plot(n_components, [m.aic(df_ssr) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.title('Bayesian information criterion (BIC)')
    plt.xlabel('n_components')
    plt.show()


def train_mlp(test, input_df: pd.DataFrame, train_x_columns: list, train_y_columns: str,
              n_layers: int, n_neurons: int, max_iter: int,
              scaler_output: str, model_prefix: str):
    """
    train a mlp model
    :param train_y_columns:
    :param train_x_columns:
    :param model_prefix: output trained model name prefix
    :param scaler_output:
    :param test:
    :param input_df:
    :param train_x: columns of the input csv file
    :param train_y:
    :param n_layers:
    :param n_neurons:
    :param max_iter:
    :return:
    """
    if test:
        print(f'start training in {n_layers:d} layers x {n_neurons:d}')

    # history = read_training_data(input_file, train_columns=train_x_columns + [train_y_columns])
    #
    # if test:
    #     print(f'reading done ..., start training...')

    # history.to_parquet('training_data.parquet')
    # history = pq.read_table(input_file).to_pandas()

    train_set = input_df

    # train_point = train_set.__len__()

    train_y_history = train_set[[train_y_columns]]
    train_x_history = train_set[train_x_columns]

    # vars into the model:
    train_x = np.atleast_2d(train_x_history)
    train_y = train_y_history.values.ravel()

    x_train = scaler_standard(fitted_array=train_x, input_array=train_x, output=scaler_output)

    # # now apply the transformations to the data:
    # x_train = scaler.transform(train_x)
    y_train = train_y

    hidden_layer_sizes = tuple(np.full((n_layers,), n_neurons))
    mlp = MLPRegressor(tol=0.00001, n_iter_no_change=200,
                       hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, shuffle=True)

    mlp.fit(x_train, y_train)

    score = mlp.score(x_train, y_train)
    # returns the coefficient of determination r^2 of the prediction.
    # the coefficient r^2 is defined as (1 - u/v),
    # where u is the residual sum of squares ((y_true - y_pred) ** 2).sum()
    # and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
    # the best possible score is 1.0 and it can be negative
    # (because the model can be arbitrarily worse).
    # a constant model that always predicts the expected value of y,
    # disregarding the input features, would get a r^2 score of 0.0.

    trained_model = f'{model_prefix:s}_{n_layers:d}_layers_{n_neurons:d}_neurons_training_days_r2_{score:.3f}.sav'

    pickle.dump(mlp, open(trained_model, 'wb'))
    if test:
        print(f'done.\nmodel trained and saved in {trained_model:s}')

    return trained_model, score


def train_mlp2(test, input_df: pd.DataFrame, train_x_columns: list, train_y_columns: str,
               hidden_layer_sizes, max_iter: int,
               scaler_output: str, model_prefix: str):
    """
    train a mlp model
    :param train_y_columns:
    :param train_x_columns:
    :param model_prefix: output trained model name prefix
    :param scaler_output:
    :param test:
    :param input_df:
    :param train_x: columns of the input csv file
    :param train_y:
    :param n_layers:
    :param n_neurons:
    :param max_iter:
    :return:
    """
    # history = read_training_data(input_file, train_columns=train_x_columns + [train_y_columns])
    #
    # if test:
    #     print(f'reading done ..., start training...')

    # history.to_parquet('training_data.parquet')
    # history = pq.read_table(input_file).to_pandas()

    train_set = input_df

    # train_point = train_set.__len__()

    train_y_history = train_set[[train_y_columns]]
    train_x_history = train_set[train_x_columns]

    # vars into the model:
    train_x = np.atleast_2d(train_x_history)
    train_y = train_y_history.values.ravel()

    x_train = scaler_standard(fitted_array=train_x, input_array=train_x, output=scaler_output)

    # # now apply the transformations to the data:
    # x_train = scaler.transform(train_x)
    y_train = train_y

    # hidden_layer_sizes = tuple(np.full((n_layers,), n_neurons))
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, shuffle=True)

    mlp.fit(x_train, y_train)

    score = mlp.score(x_train, y_train)
    # returns the coefficient of determination r^2 of the prediction.
    # the coefficient r^2 is defined as (1 - u/v),
    # where u is the residual sum of squares ((y_true - y_pred) ** 2).sum()
    # and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
    # the best possible score is 1.0 and it can be negative
    # (because the model can be arbitrarily worse).
    # a constant model that always predicts the expected value of y,
    # disregarding the input features, would get a r^2 score of 0.0.

    trained_model = f'{model_prefix:s}_2_layers_x_neurons_training_days_r2_{score:.3f}.sav'

    pickle.dump(mlp, open(trained_model, 'wb'))
    if test:
        print(f'done.\nmodel trained and saved in {trained_model:s}')

    return trained_model, score


def predict_mlp(test, pred_x_df: pd.DataFrame, train_x_columns: list, train_y_columns: str,
                trained_mod: str, scaler_output: str):
    """
    make prediction of PV output based on temp_pred from meteo france.
    :param test:
    :param scaler_output:
    :param train_y_columns:
    :param train_x_columns:
    :param pred_x_df:
    :param trained_mod:
    :param temp_pred: only one time step
    :param ssr_pred: dataframe
    :return:
    """

    # load the model from disk
    mlp = pickle.load(open(trained_mod, 'rb'))

    # construct predict input df:
    pred_x: pd.DataFrame = construct_dataframe(pred_x_df, train_x_columns)

    # load and apply scale
    pred_x_2d = np.float64(np.atleast_2d(pred_x))
    scaler = joblib.load(scaler_output)
    x_pred = scaler.transform(pred_x_2d)

    # predict:
    y_pred = mlp.predict(x_pred)

    # return a pd.DataFrame
    y_pred_df = pd.DataFrame(index=pred_x.index)
    y_pred_df[train_y_columns] = y_pred

    return y_pred_df


def print_output(df, columns, prefix, output_var_name):

    for i in range(df.__len__()):

        output = f'{prefix:s} date_concerned="{str(df.index.to_pydatetime()[i]):s}",' \
            f'{output_var_name:s}={df[columns][i]:0.3f} ' \
            f'{float(datetime.datetime.timestamp(df.index.to_pydatetime()[i])) * 1000000000:.0f}'
        print(output)


def bic_test(X, y):

    from sklearn.preprocessing import MinMaxScaler
    from RegscorePy import bic
    # https://pypi.org/project/RegscorePy/

    # testing the model in 2 hidden layers by BIC:
    # ----------------------------- model setup -----------------------------
    params = {'solver': 'adam', 'learning_rate_init': 0.01, 'nesterovs_momentum': False,
              'activation': 'tanh', 'batch_size': 200, 'max_iter': 1000, 'tol': 0.001,
              'n_iter_no_change': 100, 'verbose': 1, 'random_state': None}

    # ----------------------------- bic parameters -----------------------------
    max_neuron = 20
    layer1 = range(2, max_neuron)
    layer2 = range(2, max_neuron)

    bic_2D = np.zeros((len(layer1), len(layer2)))
    # ----------------------------- loop in topology -----------------------------
    for n1 in layer1:
        for n2 in layer2:
            print(f'reading done ..., start training...')

            hidden_layer_sizes = tuple((n1, n2))

            # ----------------------------- training: -----------------------------
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, **params)

            mlp.fit(X, y)
            print(f'training score: {mlp.score(X, y):4.2f}')

            y_pred = mlp.predict(X)
            y_true = y

            # ----------------------------- cal the BIC -----------------------------
            n_input = X.shape[1]
            n_output = 1

            n_coefs = n_input + (n1 + n2) + n_output
            n_intercepts = n1 + n2 + n_output
            p = n_coefs + n_intercepts

            bic_value = bic.bic(y_true, y_pred, p)

            bic_2D[layer1.index(n1), layer2.index(n2)] = round(bic_value, 0)

            print(f'n1 = {n1:d}, n2 = {n2:d}, p = {p:d}, bic_value = {bic_value:4.2f}')

    print(f'bic calculation is done')

    # .---------------------------- plotting -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w',
                           edgecolor='k', sharex='none', sharey='none', dpi=200)

    delta_bic = bic_2D - bic_2D.min()
    plt.imshow(delta_bic, cmap=plt.get_cmap("YlGn", 20))
    plt.colorbar()

    x = np.arange(2, max_neuron, 1)  # the grid to which your data corresponds
    nx = x.shape[0]
    no_labels = 9  # how many labels to see on axis x
    step_x = int(nx / (no_labels - 1))  # step between consecutive labels
    x_positions = np.arange(0, nx, step_x)  # pixel count at label position
    x_labels = x[::step_x]  # labels you want to see
    plt.xticks(x_positions, x_labels)
    plt.yticks(x_positions, x_labels)

    plt.xlabel(f'n_neuron in hidden layer 1', fontsize=16)
    plt.ylabel(f'n_neuron in hidden layer 2', fontsize=16)

    plt.title(f'\u0394(BIC)')

    plt.show()
    print(f'done')


def best_trained_mlp(train_X, train_Y, valid_X, valid_Y, loop: int):
    """
    training the mlp 'loop' times, give the one with best R^2 and RMSE with valid_set
    :param train_set:
    :param valid_set:
    :param loop:
    :return: model, R^2 and nMAE
    """

    import time

    # ----------------------------- model setup -----------------------------
    params = {'solver': 'adam', 'learning_rate_init': 0.1, 'nesterovs_momentum': False,
              'activation': 'tanh', 'batch_size': 200, 'max_iter': 5000, 'tol': 0.001,
              'hidden_layer_sizes': (6, 4), 'n_iter_no_change': 500, 'verbose': 1, 'random_state': None}

    best_model_output = f'best_mlp_3x21.sav'
    # ----------------------------- data set -----------------------------
    X_train = train_X
    X_valid = valid_X

    y_train = train_Y
    y_valid = valid_Y
    # ----------------------------- loop in topology -----------------------------

    mlps = []
    nmaes = []
    train_scores = []
    for i in range(loop):
        # ----------------------------- training: -----------------------------
        mlp: MLPRegressor = MLPRegressor(**params)

        mlp.fit(train_X, y_train)
        train_score = mlp.score(train_X, y_train)

        y_pred = mlp.predict(valid_X)
        y_true = y_valid

        rmse = np.sqrt(mean_squared_error(y_pred, y_true))
        nrmse = rmse/(y_true.max() - y_true.min())*100

        mae = mean_absolute_error(y_pred, y_true)
        nmae = mae/(y_true.max() - y_true.min())*100

        mlps.append(mlp)
        nmaes.append(nmae)
        train_scores.append(train_score)

        print(f'{i:g}, training score = {train_score:4.2f}, nMAE = {nmae:4.2f}')

        # -----------------------------sleep for new random seed -----------------------------
        time.sleep(5)

    # ----------------------------- find the best model -----------------------------
    best = nmaes.index(min(nmaes))

    pickle.dump(mlps[best], open(best_model_output, 'wb'))

    return best_model_output, train_scores[best], nmaes[best]

