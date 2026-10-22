import pandas as pd
import numpy as np
import pytest
import sys
sys.path.append('src')  # Add the src directory to the path

from preprocess import validate_dataframe, clean_split_data, impute_scale_data

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Pregnancies': [1, 2, 3, 14, 3],
        'Glucose': [20, 100, 141, 0, 96],
        'BloodPressure': [65, 0, 95, 50, 60],
        'SkinThickness': [25, 30, 50, 90, 35],
        'Insulin': [0, 35, 50, 700, 300],
        'BMI': [30.2, 40, 45.3, 67, 50.2],
        'DiabetesPedigreeFunction': [0.09, 0.25, 0.30, 1.56, 0.5],
        'Age': [22, 0, 40, 90, 85],
        'Outcome': [1, 0, 1, 0, 0]
    })


class TestValidateDataframe:
    def test_validate_dataframe_success(self, sample_dataframe):
        required_columns = ['Pregnancies', 'BloodPressure', 'Insulin', 'BMI', 'Outcome']
        target_column = 'Outcome'
        assert validate_dataframe(sample_dataframe, required_columns, target_column) == True

    def test_validate_dataframe_missing_column(self, sample_dataframe):
        with pytest.raises(ValueError, match="The DataFrame is missing required columns*"):
            required_columns = ['Age', 'missing_column']
            target_column = 'Outcome'
            validate_dataframe(sample_dataframe, required_columns, target_column)

    def test_validate_dataframe_missing_target(self, sample_dataframe):
        with pytest.raises(ValueError, match="The target column *"):
            required_columns = ['Age', 'Insulin']
            target_column = 'MISSING_TARGET'
            validate_dataframe(sample_dataframe, required_columns, target_column)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame({'Age':[], 'Outcome':[]})
        with pytest.raises(ValueError, match="The DataFrame is empty."):
            required_columns = ['Age']
            target_column = 'Outcome'
            validate_dataframe(empty_df, required_columns, target_column)

class TestCleanDataframe:
    def test_fills_zeros(self, sample_dataframe):
        required_columns = ['Pregnancies', 'BloodPressure', 'Insulin', 'BMI', 'Age']

        x_train, x_test, _, _ = clean_split_data(sample_dataframe, required_columns, 'Outcome', 0.4, 42)
        X_train, X_test, _, _ = impute_scale_data(x_train, x_test)
        
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_does_not_modify_original(self, sample_dataframe):
        original = sample_dataframe.copy(deep=True)

        clean_split_data(
            sample_dataframe,
            feature_columns=["Glucose", "BMI"],
            target_column="Outcome",
            test_size=0.4,
            random_state=42
        )

        pd.testing.assert_frame_equal(sample_dataframe, original)

    def test_fills_with_median_for_zeros(self, sample_dataframe):
        sample = sample_dataframe.drop(columns=['Outcome']).copy()

        x_train = sample.copy()
        x_test = sample.copy()

        _, _, imputer, _ = impute_scale_data(x_train, x_test)

        expected_median = x_train['Glucose'].median()

        glucose_idx = x_train.columns.get_loc('Glucose')

        assert imputer.statistics_[glucose_idx] == expected_median

    def test_scale_numerics_to_expected_ranges(self, sample_dataframe):
        sample = sample_dataframe.drop(columns=["Outcome"])

        X_train, _, _, _ = impute_scale_data(sample, sample.copy())

        # Check means are approximately 0
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-7)

        # Check standard deviations are approximately 1
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-7)