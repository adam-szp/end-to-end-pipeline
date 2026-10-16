from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from train import train_model
from utils import load_config
import mlflow

config = load_config()
model, X_test_scaled, y_test, run_id = train_model(config)

def evaluate_model(model, X_test_scaled, y_test, config, run_id):

    with mlflow.start_run(run_id=run_id):
        y_pred = model.predict(X_test_scaled)
        if 'accuracy' in config['evaluation']['metrics']:
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric('accuracy', round(accuracy, 4))
        if 'precision' in config['evaluation']['metrics']:
            precision = precision_score(y_test, y_pred, average='weighted')
            mlflow.log_metric('precision', round(precision, 4))
        if 'recall' in config['evaluation']['metrics']:
            recall = recall_score(y_test, y_pred, average='weighted')
            mlflow.log_metric('recall', round(recall, 4))
        if 'f1' in config['evaluation']['metrics']:
            f1 = f1_score(y_test, y_pred, average='weighted')
            mlflow.log_metric('f1', round(f1, 4))
        if 'auc_roc' in config['evaluation']['metrics']:
            auc_roc = roc_auc_score(y_test, y_pred, average='weighted')
            mlflow.log_metric('auc_roc', round(auc_roc, 4))

        print(f"Model Evaluation:\nAccuracy: {round(accuracy, 4)}\nPrecision: {round(precision, 4)}\nRecall: {round(recall, 4)}\nF1 Score: {round(f1, 4)}\nAUC ROC: {round(auc_roc, 4)}")

        if f1 < config['evaluation']['min_f1']:
            print(f"Model does not meet the F1 score threshold of {config['evaluation']['min_f1']}.")
            raise SystemExit(1)
        if precision < config['evaluation']['min_precision']:
            print(f"Model does not meet the precision threshold of {config['evaluation']['min_precision']}.")
            raise SystemExit(1)
        if accuracy < config['evaluation']['min_accuracy']:
            print(f"Model does not meet the accuracy threshold of {config['evaluation']['min_accuracy']}.")
            raise SystemExit(1)
    return accuracy, precision, recall, f1, auc_roc
