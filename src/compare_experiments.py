import mlflow
from utils import load_config

config = load_config()

experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["metrics.recall DESC"]
)

print("Top 5 runs based on Recall:")
for index, row in runs.head(5).iterrows():
    print(f"\nRun: {row['run_id'][:8]}...")
    print(f"  Model Type: {row['params.model_type']}")
    if 'recall' in config['evaluation']['metrics']:
        print(f"  Recall: {row['metrics.recall']:.4f}")
    if 'precision' in config['evaluation']['metrics']:
        print(f"  Precision: {row['metrics.precision']:.4f}")
    if 'f1_score' in config['evaluation']['metrics']:
        print(f"  F1 Score: {row['metrics.f1']:.4f}")
    if 'accuracy' in config['evaluation']['metrics']:
        print(f"  Accuracy: {row['metrics.accuracy']:.4f}")
    if 'auc_roc' in config['evaluation']['metrics']:
        print(f"  AUC ROC: {row['metrics.auc_roc']:.4f}")

best_run = runs.iloc[0]
print(f"\nBest Run: {best_run['run_id'][:8]}...")
print(f"  Run Name: {best_run['tags.mlflow.runName']}")
print(f"  Model Type: {best_run['params.model_type']}")
if 'recall' in config['evaluation']['metrics']:
    print(f"  Recall: {best_run['metrics.recall']:.4f}")
if 'precision' in config['evaluation']['metrics']:
    print(f"  Precision: {best_run['metrics.precision']:.4f}")
if 'f1_score' in config['evaluation']['metrics']:
    print(f"  F1 Score: {best_run['metrics.f1']:.4f}")
if 'accuracy' in config['evaluation']['metrics']:
    print(f"  Accuracy: {best_run['metrics.accuracy']:.4f}")
if 'auc_roc' in config['evaluation']['metrics']:
    print(f"  AUC ROC: {best_run['metrics.auc_roc']:.4f}")