import pandas as pd
import mlflow
import mlflow.sklearn
import sys
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from utils import load_config, load_hash
from preprocess import validate_dataframe, clean_dataframe, encode_categoricals, check_data_quality
from evaluate import evaluate_model

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

        df = load_data(config['data']['raw_path'])

        required = [config['data']['target_column']] + config['data']['numeric_columns'] + config['data']['categorical_columns']
        to_drop = [col for col in df.columns if col not in required]
        df.drop(columns=to_drop, inplace=True)
        
        if not validate_dataframe(df, required, config['data']['target_column']):
            sys.exit(1)

        quality = check_data_quality(df, config['data']['numeric_columns'])
        print (f"Data Quality: {quality['total_nulls']} nulls, {quality['total_duplicates']} duplicates")
        
        df = clean_dataframe(df, config['data']['numeric_columns'], config['data']['categorical_columns'], config['data']['date_columns'], config['data']['target_column'])
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
        mlflow.log_param('features_dropped', str(to_drop))

        if config["model"]["model_type"] == "RandomForestClassifier":
            trusted_types = ["sklearn.tree._tree.Tree"]
        else:
            trusted_types = None
        
        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=trusted_types
        )

        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)

        hash_value = load_hash()
        mlflow.log_param('dvc_data_hash', hash_value)

        run_id = mlflow.active_run().info.run_id
        print(f"MLFlow Run ID: {run_id}")

    return model, X_test_scaled, y_test, run_id

def main():
    """
    Main function to execute the training pipeline.
    """
    try:
        model, X_test_scaled, y_test, run_id = train_model(config)
        print("Model training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)

    with mlflow.start_run(run_id=run_id):
        metrics = evaluate_model(model, X_test_scaled, y_test)
        
        if 'accuracy' in config['evaluation']['metrics']:
            accuracy = metrics['accuracy']
            mlflow.log_metric('accuracy', round(accuracy, 4))
        if 'precision' in config['evaluation']['metrics']:
            precision = metrics['precision']
            mlflow.log_metric('precision', round(precision, 4))
        if 'recall' in config['evaluation']['metrics']:
            recall = metrics['recall']
            mlflow.log_metric('recall', round(recall, 4))
        if 'f1_score' in config['evaluation']['metrics']:
            f1 = metrics['f1_score']
            mlflow.log_metric('f1', round(f1, 4))
        if 'auc_roc' in config['evaluation']['metrics']:
            auc_roc = metrics['roc_auc']
            mlflow.log_metric('auc_roc', round(auc_roc, 4))


        if f1 < config['evaluation']['min_f1']:
            print(f"Model does not meet the F1 score threshold of {config['evaluation']['min_f1']}.")
            raise SystemExit(1)
        if precision < config['evaluation']['min_precision']:
            print(f"Model does not meet the precision threshold of {config['evaluation']['min_precision']}.")
            raise SystemExit(1)
        if accuracy < config['evaluation']['min_accuracy']:
            print(f"Model does not meet the accuracy threshold of {config['evaluation']['min_accuracy']}.")
            raise SystemExit(1)
        if recall < config['evaluation']['min_recall']:
            print(f"Model does not meet the recall threshold of {config['evaluation']['min_recall']}.")
            raise SystemExit(1)
        if auc_roc < config['evaluation']['min_auc_roc']:
            print(f"Model does not meet the AUC ROC threshold of {config['evaluation']['min_auc_roc']}.")
            raise SystemExit(1)

        print("Model evaluation metrics logged successfully.")

if __name__ == "__main__":
    main()