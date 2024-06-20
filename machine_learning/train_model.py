"""
Author: jhzhu
Date: 2024/6/20
Description: 
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def custom_adjusted_r2(y_true, y_pred, **kwargs):
    if 'x_column' not in kwargs['kwargs']:
        return 0
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    return 1 - (1 - r2) * (len(y_pred) - 1) / (len(y_pred) - kwargs['kwargs']['x_column'] - 1)


def custom_error_percentage(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        return round(1 - np.sum(np.abs(y_true.values.flatten() - y_pred)) / np.sum(y_true), 3)[0]
    else:
        return round(1 - np.sum(np.abs(y_true - y_pred)) / np.sum(y_true), 3)


def custom_error_percentage_avg(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    return np.round(np.mean(1 - np.abs(y_true - y_pred) / y_true), 3)


def evaluate_predict_result(x, y_true, y_pred):
    result_dict = dict()
    result_dict['mean_absolute_error'] = round(mean_absolute_error(y_true=y_true, y_pred=y_pred), 3)
    result_dict['median_absolute_error'] = round(median_absolute_error(y_true=y_true, y_pred=y_pred), 3)
    result_dict['root_mean_squared_error'] = round(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 3)
    result_dict['r2'] = round(r2_score(y_true=y_true, y_pred=y_pred), 3)
    result_dict['adjusted_r2'] = round(custom_adjusted_r2(y_true, y_pred, kwargs={'x_column': x.shape[1]}), 3)
    result_dict['accuracy'] = round(custom_error_percentage(y_true, y_pred), 3)
    result_dict['avg accuracy'] = round(custom_error_percentage_avg(y_true, y_pred), 3)
    return result_dict


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    result = evaluate_predict_result(X_test, y_test, rf_pred)
    print(result)
