import pytest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from app import extract_features, PatientData


class TestInterface:

    def test_extract_features(self):
        mock_llm = Mock()

        sample_input = "I'm 35, my glucose is 148 and BMI is 33.6."

        llm_response = """{
            "Pregnancies": null,
            "Glucose": 148,
            "BloodPressure": null,
            "SkinThickness": null,
            "Insulin": null,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": null,
            "Age": 35,
            "in_scope": true
        }"""

        with patch("app.ask_llm", return_value=llm_response):
            result = extract_features(mock_llm, sample_input)

        assert result["Age"] == 35
        assert result["Glucose"] == 148
        assert result["BMI"] == 33.6
        assert result["Pregnancies"] is None

    def test_incomplete_input(self):
        mock_llm = Mock()

        llm_response = """{
            "Pregnancies": null,
            "Glucose": 120,
            "BloodPressure": null,
            "SkinThickness": null,
            "Insulin": null,
            "BMI": null,
            "DiabetesPedigreeFunction": null,
            "Age": 40,
            "in_scope": true
        }"""

        with patch("app.ask_llm", return_value=llm_response):
            result = extract_features(
                mock_llm,
                "I'm 40 and my glucose is 120."
            )

        missing = [
            feature for feature, value in result.items()
            if value is None
        ]

        assert result["Age"] == 40
        assert result["Glucose"] == 120
        assert len(missing) == 6

    def test_invalid_input(self):
        mock_llm = Mock()

        llm_response = """{
            "Pregnancies": null,
            "Glucose": "not a number",
            "BloodPressure": null,
            "SkinThickness": null,
            "Insulin": null,
            "BMI": null,
            "DiabetesPedigreeFunction": null,
            "Age": 35,
            "in_scope": true
        }"""

        with patch("app.ask_llm", return_value=llm_response):
            with pytest.raises(ValueError):
                extract_features(
                    mock_llm,
                    "I'm 35 but I don't know my glucose."
                )