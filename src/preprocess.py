import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# My chosen dataset does not have any categorical columns so there is no need to create an encode_categorical function.

def validate_dataframe(df, required_columns, target_column ) -> bool:
    """
    Validate the input DataFrame to ensure it meets the required criteria.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate.

    Returns:
    bool: True if the DataFrame is valid, False otherwise.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    if df.isnull().values.any():
        print("The DataFrame contains missing values.")

    if df.duplicated().any():
        print("The DataFrame contains duplicate rows, will drop duplicates.")
        df.drop_duplicates(inplace=True)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"The DataFrame is missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"The target column '{target_column}' is not in the DataFrame")


    return True

def clean_split_data(df, feature_columns, target_column, test_size, random_state):
    """
    Clean the input DataFrame by handling missing values and split the data using train_test_split.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    feature_columns (list): List of feature column names.
    target_column (str): The name of the target column.
    test_size (float): Test size percentage.
    random_state (int): Random state int.
    stratify (str): Whether to stratify the split data.

    Returns:
    train_test_split: The cleaned and split features and targets.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
       
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    features = df.drop(columns=[target_column])
    target = df[target_column]

    return train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target)

def impute_scale_data(x_train, x_test):
    """
    Impute and then scale only on train data to prevent data leakage.

    Parameters:
    x_train (pd.Dataframe): Training Dataframe.
    x_test (pd.Dataframe): Test Dataframe.

    Returns:
    X_train (pd.Dataframe): Good Training Dataframe.
    X_test (pd.Dataframe): Good Test Dataframe.
    imputer: The fitted imputer to log later.
    scaler: The fitted scaler to log later.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(x_train)
    X_test = imputer.transform(x_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, imputer, scaler

def check_data_quality(df, numeric_columns):
    """
    Check the quality of the data in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    numeric_columns (list): List of numeric column names.

    Returns:
    dict: A dictionary containing data quality metrics.
    """
    report = {
        'total_rows': len(df),
        'total_nulls': int(df.isnull().sum().sum()),
        'null_percentage': round(df.isnull().sum().sum()/ (len(df) * len(df.columns)) * 100, 2),
        'total_duplicates': int(df.duplicated().sum())
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())
            
    return report