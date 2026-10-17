import os
import sys
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

from utils import load_config
from preprocess import clean_dataframe

config = load_config()

def load_reference_data():
    df = pd.read_csv(config['data']['raw_path'])

    df = clean_dataframe(df, config['data']['numeric_columns'], config['data']['categorical_columns'], config['data']['date_columns'], config['data']['target_column'])

    return df

def generate_production_data(reference):
    production = reference.sample(frac=0.3, random_state=42).copy()

    if 'age' in production.columns:
        production['age'] = production['age'] + 5

    if 'length_of_service' in production.columns:
        production['length_of_service'] = production['length_of_service'] * 1.25

    return production

def main():
    reference = load_reference_data()

    target = config['data']['target_column']

    if target in reference.columns:
        reference_features = reference.drop(columns=[target])
    else:
        reference_features = reference

    production = generate_production_data(reference_features)

    report = Report(metrics = [DataDriftPreset()])

    snapshot = report.run(reference_data = reference_features, current_data = production)

    result = snapshot.dict()
    drift_count_metric = result['metrics'][0]
    drifted_count = int(drift_count_metric['value']['count'])
    drift_share = drift_count_metric['value']['share']

    feature_metrics = result['metrics'][1:]
    total_features = len(feature_metrics)

    drift_threshold = config['monitoring']['drift_threshold']

    summary = {
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 3),
        "dataset_drift": drift_share >= drift_threshold,
    }

    feature_details = {}
    drifted_features = []
    for metric in feature_metrics:
        column = metric['config']['column']
        threshold = metric['config']['threshold']
        drift_value = float(metric['value'])

        drifted = drift_value >= threshold

        if drifted:
            drifted_features.append(column)

        feature_details[column] = {
            'drifted': drift_value >= threshold,
            'threshold': threshold,
            'drift_score': round(drift_value, 4),
            'method': metric['config']['method']
        }    

    summary['features'] = feature_details

    print(f'Total features: {total_features}')
    print(f'Drifted features: {drifted_count}')
    print(f'Drift share: {drift_share:.3f}')
    print(f'Drift threshold: {drift_threshold:.3f}')

    print('\nFeatures with drift:')
    for feature in drifted_features:
        print(f'- {feature}')

    os.makedirs('reports', exist_ok=True)

    report_path = config['monitoring']['report_path']
    snapshot.save_html(report_path)

    print(f'Drift report saved to: {report_path}')

    if drift_share > drift_threshold:
        print('Drift threshold exceeded.')
        sys.exit(1)

    print('Drift is within acceptable limits.')


if __name__ == '__main__':
    main()