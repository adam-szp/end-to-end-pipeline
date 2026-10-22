import pandas as pd
import pytest
import sys

sys.path.insert(0, 'src')
from utils import load_config

config = load_config()

@pytest.fixture
def sample_data():
    df = pd.read_csv(config['data']['raw_path'])
    return df.sample(n=50, random_state=42)

class TestDataValidate:
    def test_expected_columns_exist(self, sample_data):
        required = [config['data']['target_column']] + config['data']['numeric_columns'] + config['data']['categorical_columns']
        assert set(required).issubset(set(sample_data.columns))

    def test_target_values(self, sample_data):
        target = config['data']['target_column']
        assert set(sample_data[target].unique()) <= {0, 1}

    def test_numeric_range(self, sample_data):
        assert sample_data['Age'].between(20, 82).all()
        assert sample_data['DiabetesPedigreeFunction'].between(0.07, 2.43).all()
