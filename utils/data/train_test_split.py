
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_df(dataframe: pd.DataFrame):
    # Create a new column for binned 'Keff'
    dataframe['Keff_bins'] = pd.cut(dataframe['Keff'], bins=15, labels=False)

    # Use 'Keff_bins' for stratification
    train_df, test_df = train_test_split(dataframe, test_size=0.2,
                                         random_state=9320, stratify=dataframe['Keff_bins'])

    # Remove the 'Keff_bins' column from the train and test sets
    train_df = train_df.drop(columns=['Keff_bins'])
    test_df = test_df.drop(columns=['Keff_bins'])

    return train_df, test_df
