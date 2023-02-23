import os.path
import pickle
from ctypes import Union

import numpy as np
import pandas as pd


def _load_feature_value(pkl_data_path: str) -> (np.ndarray, np.ndarray):
    with open(pkl_data_path, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']
        return X, y


def get_data(cache_dir, input_file, target_name, time_length: int, predict_length: int) -> (np.ndarray, np.ndarray):
    """
    preprocessing data from original data and return the numpy result
    if data is preprocessed before, return the previously stored result
    :param predict_length: predicting future time from current time
    :param target_name: name of target value in CSV file
    :param input_file: input CSV data file
    :param cache_dir: middle cache dir to store arranged data
    :param time_length: length of time in feature
    :return: X,y is feature, target.
    """
    pkl_data_path = os.path.join(cache_dir, os.path.split(input_file)[-1].replace('.csv', f'_{time_length}.pkl'))
    if os.path.exists(pkl_data_path):
        return _load_feature_value(pkl_data_path)
    data = pd.read_csv(input_file)

    # check target_name exist
    if not target_name in data.columns:
        raise RuntimeError(f'not column named {target_name} in input file {input_file}')

    # convert all data to float values, (make sure step)
    try:
        for column in data.columns:
            data.loc[:, column] = data.loc[:, column].apply(
                lambda x: float(x))
    except:
        raise RuntimeError(f'not all data be float value, please check again or use custom data processing method')

    X, y = [], []
    for i in range(time_length, data.shape[0] - predict_length):
        X.append(data.loc[i - time_length:i - 1, :].to_numpy())
        y.append(data.loc[i + predict_length - 1, target_name])
    X = np.array(X)
    y = np.array(y)
    with open(pkl_data_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    return _load_feature_value(pkl_data_path)


def get_predict_data(input_file, target_name, time_length: int) -> (np.ndarray, np.ndarray):
    """
    preprocessing data from original data and return the numpy result of feature
    if data is preprocessed before, return the previously stored result
    :param target_name: name of target value in CSV file
    :param input_file: input CSV data file
    :param time_length: length of time in feature
    :return: X (feature)
    """

    data = pd.read_csv(input_file)

    # check target_name exist
    if not target_name in data.columns:
        raise RuntimeError(f'not column named {target_name} in input file {input_file}')

    # convert all data to float values, (make sure step)
    try:
        for column in data.columns:
            data.loc[:, column] = data.loc[:, column].apply(
                lambda x: float(x))
    except:
        raise RuntimeError(f'not all data be float value, please check again or use custom data processing method')

    X = []
    for i in range(time_length, data.shape[0] +1 ):
        X.append(data.loc[i - time_length:i - 1, :].to_numpy())
    X = np.array(X)
    return X


def train_test_split(x, y, test_ratio_or_size):
    """
    If test_ratio is float in (0,1), split the whole dataset into train and test datasets and test dataset contains test_ration of whole dataset.
    else test_ratio represent absolute size of test dataset
    :param x: feature data
    :param y: target data
    :param test_ratio_or_size: float value in range (0, 1), or int value of test dataset size
    :return: X_train, y_train, X_test, y_test
    """
    length = x.shape[0]
    train_length = int(length * (1 - test_ratio_or_size)) if isinstance(test_ratio_or_size, float) else length - test_ratio_or_size
    X_train, y_train, X_test, y_test = x[:train_length], y[:train_length], x[train_length:], y[train_length:]
    return X_train, y_train, X_test, y_test
