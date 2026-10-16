import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import sys
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_config, load_hash
from preprocess import validate_dataframe, clean_dataframe, encode_categoricals, check_data_quality

config = load_config()

def load_data(path):
    """
    Load data from a CSV file.

    Parameters:
    path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def train_model(config):
    """
    Train a Random Forest model using the provided configuration.

    Parameters:
    config (dict): Configuration dictionary containing paths and parameters.
    """

    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():

        df = load_data(config['data']['path'])

        useless = ['EmployeeID' ,'recorddate_key', 'gender_full']
        for col in useless:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        required = config['data']['target_column'] + config['data']['numeric_columns'] + config['data']['categorical_columns']
        if not validate_dataframe(df, required, config['data']['target_column']):
            sys.exit(1)

        quality = check_data_quality(df, config['data']['numeric_columns'])
        print (f"Data Quality: {quality['total_nulls']} nulls, {quality['total_duplicates']} duplicates")
        
        df = clean_dataframe(df, config['data']['numeric_columns'], config['data']['categorical_columns'])
        df = encode_categoricals(df, config['data']['categorical_columns'])

        
        X = df.drop(columns=[config['data']['target_column']])
        y = df[config['data']['target_column']]

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state'])

        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_options = {
            'RandomForestClassifier': RandomForestClassifier,
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression
        }
        model_class = model_options[config['model']['model_type']]
        model = model_class(**config['model']['parameters'])
        model.fit(X_train_scaled, y_train)

        mlflow.log_param('model_type', config['model']['model_type'])
        mlflow.log_params(config['model']['parameters'])
        mlflow.log_param('features_dropped', str(useless))

        mlflow.sklearn.log_model(model, "model")

        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)

        hash_value = load_hash()
        mlflow.log_param('dvc_data_hash', hash_value)

        run_id = mlflow.active_run().info.run_id
        print(f"MLFlow Run ID: {run_id}")
        print('Model training completed and logged to MLflow.')

    return model, X_test_scaled, y_test, run_id