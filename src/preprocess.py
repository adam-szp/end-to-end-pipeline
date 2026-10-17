import pandas as pd

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

def clean_dataframe(df, numeric_columns, categorical_columns, date_columns, target_column):
    """
    Clean the input DataFrame by handling missing values, as well as cleaning date columns into just the year and mapping target column to binary.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    numeric_columns (list): List of numeric column names.
    categorical_columns (list): List of categorical column names.
    date_columns (list): List of date column names.
    target_column (str): The name of the target column.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    for col in numeric_columns:
        if col in df.columns:
            df[col] =df[col].fillna(df[col].mean())

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.year

    if target_column in df.columns:
        df[target_column] = df[target_column].map({'ACTIVE': 0, 'TERMINATED': 1})
    else:
        print('Target column not in Dataframe')

    return df

def encode_categoricals(df, categorical_columns):
    """
    Encode categorical columns in the DataFrame using One-hot Encoding.

    Parameters:
    df (pd.DataFrame): The DataFrame containing categorical columns.
    categorical_columns (list): List of categorical column names to encode.

    Returns:
    pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

    return df

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