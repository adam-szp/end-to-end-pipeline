import pandas as pd
import pytest
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
sys.path.insert(0, 'src')

from utils import load_config
from preprocess import validate_dataframe, clean_dataframe, encode_categoricals

config = load_config()

@pytest.fixture
def sample_model():
    df_raw = pd.read_csv(config['data']['raw_path'])
    df = df_raw.groupby(config['data']['target_column'], group_keys=False).sample(frac=0.05, random_state=42)

    required = [config['data']['target_column']] + config['data']['numeric_columns'] + config['data']['categorical_columns']
    to_drop = [col for col in df.columns if col not in required]
    df.drop(columns=to_drop, inplace=True)

    if not validate_dataframe(df, required, config['data']['target_column']):
        print('No target column.')
        sys.exit(1)

    df = clean_dataframe(df, config['data']['numeric_columns'], config['data']['categorical_columns'], config['data']['date_columns'], config['data']['target_column'])
    df = encode_categoricals(df, config['data']['categorical_columns'])

    X = df.drop(columns=[config['data']['target_column']])
    y = df[config['data']['target_column']]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state'], stratify=y)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

class TestModelValidate:
    def test_model_predict_shape(self, sample_model):
        model, X_test, _ = sample_model
        
        preds = model.predict(X_test)

        assert len(preds) == len(X_test)
        assert set(preds) <= {0, 1}

    def test_model_min_performance(self, sample_model):
        model, X_test, y_test = sample_model
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)

        if 'f1_score' in config['evaluation']['metrics']:
            f1 = f1_score(y_test, preds)
            assert f1 >= 0.4
        if 'accuracy' in config['evaluation']['metrics']:
            accuracy = accuracy_score(y_test, preds)
            assert accuracy >= 0.7
        if 'precision' in config['evaluation']['metrics']:
            precision = precision_score(y_test, preds)
            assert precision >= 0.35
        if 'recall' in config['evaluation']['metrics']:
            recall = recall_score(y_test, preds)
            assert recall >= 0.4
        if 'auc_roc' in config['evaluation']['metrics']:
            auc_roc = roc_auc_score(y_test, preds_proba)
            assert auc_roc >= 0.45