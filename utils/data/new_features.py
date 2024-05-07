import pandas as pd
import numpy as np


def add_all_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = add_vmvf(dataframe)
    dataframe = add_vol_uranium(dataframe)
    dataframe = add_u_vmvf(dataframe)
    dataframe = add_temp_diff(dataframe)
    dataframe = add_temp_ratio(dataframe)
    dataframe = add_fuel_dist(dataframe)
    return dataframe


def add_vmvf(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Moderator-to-fuel Ratio
    dataframe['vmvf'] = ((dataframe['Pitch'] * dataframe['Pitch']
                         / (np.pi * dataframe['FuelRad'] * dataframe['FuelRad']))
                         - 1 - (0.12 / dataframe['FuelRad']) -
                         (0.0036 / (dataframe['FuelRad'] * dataframe['FuelRad'])))
    return dataframe


def add_vol_uranium(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Volume of Uranium
    dataframe['VolUranium'] = (np.pi * dataframe['FuelRad'] * dataframe['FuelRad'])
    return dataframe


def add_u_vmvf(dataframe: pd.DataFrame) -> pd.DataFrame:
    if 'vmvf' not in dataframe.columns:
        dataframe = add_vmvf(dataframe)
    if 'VolUranium' not in dataframe.columns:
        dataframe = add_vol_uranium(dataframe)

    # Volume of Uranium to Moderator-to-fuel Ratio
    dataframe['U_vmvf'] = dataframe['VolUranium'] / dataframe['vmvf']
    return dataframe


def add_temp_diff(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Temperature Difference
    dataframe['TempDiff'] = dataframe['FuelTemp'] - dataframe['ModTemp']
    return dataframe


def add_temp_ratio(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Temperature Ratio
    dataframe['TempRatio'] = dataframe['FuelTemp'] / dataframe['ModTemp']
    return dataframe


def add_fuel_dist(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Fuel Distance
    dataframe['FuelDist'] = dataframe['Pitch'] - (dataframe['FuelRad'] * 2)
    return dataframe
