import pandas as pd
import pytest
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
sys.path.insert(0, 'src')

from utils import load_config
from preprocess import validate_dataframe, clean_split_data, impute_scale_data

config = load_config()

@pytest.fixture
def sample_model():
    
    df = pd.read_csv(config['data']['raw_path'])
    
    required = [config['data']['target_column']] + config['data']['numeric_columns'] + config['data']['categorical_columns']
    to_drop = [col for col in df.columns if col not in required]
    df.drop(columns=to_drop, inplace=True)
    
    if not validate_dataframe(df, required, config['data']['target_column']):
        sys.exit(1)

    X_train, X_test, y_train, y_test = clean_split_data(df, config['data']['numeric_columns'], config['data']['target_column'], config['data']['test_size'], config['data']['random_state'])

    X_train_scaled, X_test_scaled, _, _ = impute_scale_data(X_train, X_test)

    if config['model']['model_type'] == 'RandomForestClassifier':
        model = RandomForestClassifier(**config['model']['rf_parameters'])
    if config['model']['model_type'] == 'LogisticRegression':
        model = LogisticRegression(**config['model']['lr_parameters'])
    if config['model']['model_type'] == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**config['model']['gb_parameters'])
    
    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled, y_test

class TestModelValidate:
    def test_model_predict_shape(self, sample_model):
        model, X_test, _ = sample_model
        
        preds = model.predict(X_test)

        assert len(preds) == len(X_test)
        assert set(preds) <= {0, 1}

    def test_model_min_performance(self, sample_model):
        model, X_test, y_test = sample_model
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]

        if 'f1_score' in config['evaluation']['metrics']:
            f1 = f1_score(y_test, preds)
            assert f1 >= 0.4
        if 'accuracy' in config['evaluation']['metrics']:
            accuracy = accuracy_score(y_test, preds)
            assert accuracy >= 0.55
        if 'precision' in config['evaluation']['metrics']:
            precision = precision_score(y_test, preds)
            assert precision >= 0.35
        if 'recall' in config['evaluation']['metrics']:
            recall = recall_score(y_test, preds)
            assert recall >= 0.7
        if 'auc_roc' in config['evaluation']['metrics']:
            auc_roc = roc_auc_score(y_test, preds_proba)
            assert auc_roc >= 0.7