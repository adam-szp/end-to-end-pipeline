import pandas as pd
import numpy as np
import pytest
import sys
sys.path.append('src')  # Add the src directory to the path

from preprocess import validate_dataframe, clean_dataframe, encode_categoricals

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'age': [25, 30, np.nan, 22],
        'length_of_service': [5, 10, 3, np.nan],
        'orighiredate_key': ['1/1/2020', '05/15/2019', '07/20/2021', '03/10/2018'],
        'city_name': ['New York', 'Los Angeles', np.nan, 'Chicago'],
        'department_name': ['Sales', 'Engineering', 'HR', np.nan],
        'job_title': ['Manager', 'Engineer', 'HR Specialist', 'Analyst'],
        'store_name': [1, 2, 3, 4],
        'gender_short': ['M', 'F', 'M', np.nan],
        'STATUS_YEAR': [2006, 2007, 2008, 2009],
        'BUSINESS_UNIT': ['HEADOFFICE', 'RETAIL', 'RETAIL', 'HEADOFFICE'],
        'STATUS': ['ACTIVE', 'TERMINATED', 'ACTIVE', 'TERMINATED']
    })

class TestValidateDataframe:
    def test_validate_dataframe_success(self, sample_dataframe):
        required_columns = ['age', 'length_of_service', 'orighiredate_key', 'BUSINESS_UNIT', 'STATUS']
        target_column = 'STATUS'
        assert validate_dataframe(sample_dataframe, required_columns, target_column) == True

    def test_validate_dataframe_missing_column(self, sample_dataframe):
        with pytest.raises(ValueError, match="The DataFrame is missing required columns*"):
            required_columns = ['age', 'missing_column']
            target_column = 'STATUS'
            validate_dataframe(sample_dataframe, required_columns, target_column)

    def test_validate_dataframe_missing_target(self, sample_dataframe):
        with pytest.raises(ValueError, match="The target column *"):
            required_columns = ['age', 'length_of_service']
            target_column = 'MISSING_TARGET'
            validate_dataframe(sample_dataframe, required_columns, target_column)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame({'age':[], 'STATUS':[]})
        with pytest.raises(ValueError, match="The DataFrame is empty."):
            required_columns = ['age']
            target_column = 'STATUS'
            validate_dataframe(empty_df, required_columns, target_column)

class TestCleanDataframe:
    def test_fills_numeric_nulls(self, sample_dataframe):
        result = clean_dataframe(sample_dataframe, ['age', 'length_of_service'], ['city_name', 'job_title'], ['orighiredate_key'], 'STATUS')
        assert result['age'].isna().sum() == 0
        assert result['length_of_service'].isna().sum() == 0

    def test_does_not_modify_original(self, sample_dataframe):
        original_nulls = sample_dataframe['age'].isna().sum()
        clean_dataframe(sample_dataframe, ['age', 'length_of_service'], ['city_name', 'job_title'], ['orighiredate_key'], 'STATUS')
        assert sample_dataframe['age'].isna().sum() == original_nulls

    def test_fills_with_mean_for_numeric(self, sample_dataframe):
        result = clean_dataframe(sample_dataframe, ['age'], [], [], 'STATUS')
        expected_mean = sample_dataframe['age'].mean()
        assert result['age'].iloc[2] == expected_mean

    def test_non_null_values_unchanged(self, sample_dataframe):
        result = clean_dataframe(sample_dataframe, ['age'], [], [], 'STATUS')
        assert result['age'].iloc[0] == 25
        assert result['age'].iloc[1] == 30
        
class TestEncodeCategoricals:
    def test_creates_dummy_columns(self, sample_dataframe):
        