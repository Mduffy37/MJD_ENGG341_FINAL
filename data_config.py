import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

scaler = pickle.load(open('data_store/modelling/scaler/scaler.pkl', 'rb'))

train_df_orig = pd.read_csv('data_store/dataset/raw/train_df.csv')
test_df_orig = pd.read_csv('data_store/dataset/raw/test_df.csv')

train_df_fe = pd.read_csv('data_store/dataset/processed/train_df_fe.csv')
test_df_fe = pd.read_csv('data_store/dataset/processed/test_df_fe.csv')

open_mc_orig = pd.read_csv('data_store/dataset/openmc/openmc_data_raw.csv')
open_mc_fe = pd.read_csv('data_store/dataset/openmc/openmc_data_fe.csv')

X_train_orig = train_df_orig.drop(columns=['Keff'])
y_train_orig = train_df_orig['Keff']
X_test_orig = test_df_orig.drop(columns=['Keff'])
y_test_orig = test_df_orig['Keff']

X_train_fe = train_df_fe.drop(columns=['Keff'])
X_train_fe_scaled = scaler.transform(X_train_fe)
y_train_fe = train_df_fe['Keff']

X_test_fe = test_df_fe.drop(columns=['Keff'])
X_test_fe_scaled = scaler.transform(X_test_fe)
y_test_fe = test_df_fe['Keff']

X_open_mc_fe = open_mc_fe.drop(columns=['Keff', 'Unnamed: 0', 'Std_dev', 'time_taken'])
X_open_mc_fe_scaled = scaler.transform(X_open_mc_fe)
y_open_mc_fe = open_mc_fe['Keff']

