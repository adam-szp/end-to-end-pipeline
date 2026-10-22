import pandas as pd
import numpy as np
import pytest
import sys
sys.path.append('src')  # Add the src directory to the path

from preprocess import validate_dataframe, clean_split_data, impute_scale_data

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Pregnancies': [1, 2, 3, 14],
        'Glucose': [20, 100, 141, 0],
        'BloodPressure': [65, 0, 95, 50],
        'SkinThickness': [25, 30, 50, 90],
        'Insulin': [0, 35, 50, 700],
        'BMI': [30.2, 40, 45.3, 67],
        'DiabetesPedigreeFunction': [0.09, 0.25, 0.30, 1.56],
        'Age': [22, 0, 40, 90],
        'Outcome': [1, 0, 1, 0]
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
        x_train, x_test, _, _ = clean_split_data(sample_dataframe, ['Age', 'BloodPressure'], 'Outcome', 0.2, 42)
        X_train, X_test = impute_scale_data(x_train, x_test)
        assert X_train['Age'].isna().sum() == 0
        assert X_test['BloodPressure'].isna().sum() == 0

    def test_does_not_modify_original(self, sample_dataframe):
        original = sample_dataframe.copy(deep=True)

        clean_split_data(
            sample_dataframe,
            feature_columns=["Glucose", "BMI"],
            target_column="Outcome",
            test_size=0.2,
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