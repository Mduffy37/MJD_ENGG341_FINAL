import json
import pickle

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import optuna
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly

import utils.plotting as cplot
import utils.modelling as modelling
import utils.modelling.hyper_parameters as hp
import utils.data as data_utils
import utils.modelling.scripts as scripts

import algo_config
import data_config as data

default_algos = algo_config.default_algorithms

df = pd.read_csv('data_store/dataset/openmc/openmc_data_fe.csv')
orig_df = pd.read_csv('data_store/dataset/processed/dataset_67k_fe.csv')

df = data.open_mc_fe

modelling.ev.test_ensemble(data.X_test_fe_scaled, data.y_test_fe, 'stacked_ensemble')

y_test = data.y_open_mc_fe.values
y_pred = modelling.ev.test_final_model(data.X_open_mc_fe_scaled, y_test)

df["deltaKabs"] = np.abs(df["Keff"] - y_pred) * 100000
df["deltaKrel"] = np.abs((y_pred - df["Keff"])/df["Keff"]) * 100000

df_describe = df.describe(percentiles=[0.9, 0.95, 0.99])
orig_df_describe = orig_df.describe(percentiles=[0.9, 0.95, 0.99])

for col in orig_df.columns:
    print(f'{col}:\n Mean -> OpenMC: {df_describe[col]["mean"]}, Orig: {orig_df_describe[col]["mean"]}\n'
          f' Min -> OpenMC: {df_describe[col]["min"]}, Orig: {orig_df_describe[col]["min"]}\n'
          f' Max -> OpenMC: {df_describe[col]["max"]}, Orig: {orig_df_describe[col]["max"]}\n')

print("--------------------")

df_high_error = df[df["deltaKabs"] > 850]
print(df_high_error.index)
df_high_describe = df_high_error.describe(percentiles=[0.9, 0.95, 0.99])

df_low_error = df[df["deltaKabs"] <= 850]
df_low_describe = df_low_error.describe(percentiles=[0.9, 0.95, 0.99])

print("--------------------")

for col in df.columns:
    print(f'{col}:\n Mean -> High: {df_high_describe[col]["mean"]}, Low: {df_low_describe[col]["mean"]}\n'
          f' Min -> High: {df_high_describe[col]["min"]}, Low: {df_low_describe[col]["min"]}\n'
          f' Max -> High: {df_high_describe[col]["max"]}, Low: {df_low_describe[col]["max"]}\n')

