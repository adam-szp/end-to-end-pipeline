import math
import pandas as pd
import streamlit as st
from transformers import pipeline
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

from compare_experiments import load_best_artifacts


features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

class PatientData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pregnancies: float | None
    Glucose: float | None
    BloodPressure: float | None
    SkinThickness: float | None
    Insulin: float | None
    BMI: float | None
    DiabetesPedigreeFunction: float | None
    Age: float | None
    in_scope: bool

    @field_validator(
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    )
    @classmethod
    def validate_numbers(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Measurements must be finite numbers.")
        return value

@st.cache_resource
def load_models():
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen3-4B-Instruct-2507",
        device_map="auto",
        dtype="auto"
    )

    model, imputer, scaler = load_best_artifacts()

    return llm, model, imputer, scaler


def ask_llm(llm, system_prompt, user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = llm(
        messages,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False
    )

    result = response[0]["generated_text"]

    if isinstance(result, list):
        return result[-1]["content"]

    return result


def extract_features(llm, user_input):
    prompt = f"""
    You extract measurements for a diabetes classification model.

    Return ONLY valid JSON with these keys:
    {features}

    Also include "in_scope": true or false.

    Rules:
    - Extract only numbers the user provides.
    - Use null for missing or unclear values.
    - Never guess missing measurements.
    - Pregnancies means number of pregnancies, not children.
    - BloodPressure means diastolic blood pressure.
    - For a reading of 120/80, use 80.
    - Do not assume measurement units.
    - If values conflict, use null.
    - If the request is unrelated to diabetes classification,
    set in_scope to false.
    - Return JSON only. No explanations or markdown.
    """

    response = ask_llm(llm, prompt, user_input)

    response = response.strip()
    response = response.replace("```json", "").replace("```", "")

    try:
        data = PatientData.model_validate_json(response)
    except ValidationError as e:
        raise ValueError("Qwen returned invalid data.") from e

    if not data.in_scope:
        raise ValueError("This model only supports diabetes classification.")
    
    return data.model_dump(exclude={"in_scope"})


def predict_diabetes(data, model, imputer, scaler):
    X = pd.DataFrame([data], columns=features)

    X = imputer.transform(X)
    X = scaler.transform(X)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    return prediction, probability


def explain_prediction(llm, prediction, probability, missing):
    prompt = """
    Explain the result of an experimental diabetes classifier.

    Rules:
    - Class 1 means predicted diabetes.
    - Class 0 means predicted no diabetes.
    - Explain the result in simple language.
    - This is not a medical diagnosis.
    - The model score is not a validated medical risk.
    - Mention that the Pima dataset is small and historical.
    - Mention any measurements filled using training medians.
    - Do not invent medical causes or feature importance.
    - Keep the explanation short.
    """

    user_input = f"""
    Prediction: {prediction}
    Positive-class model score: {probability:.4f}
    Imputed measurements: {missing}
    """

    return ask_llm(llm, prompt, user_input)


st.title("Diabetes Classification Assistant")

st.write(
    "Describe your measurements and I'll use a trained "
    "machine learning model to make a classification."
)

st.caption("Educational demonstration only. Not medical advice.")


if "data" not in st.session_state:
    st.session_state.data = None

if "asked" not in st.session_state:
    st.session_state.asked = False


if st.session_state.data is None:

    with st.form("initial"):
        user_input = st.text_area(
            "Your measurements",
            placeholder="I'm 35, glucose is 148, and BMI is 33.6."
        )
        submitted = st.form_submit_button("Analyze")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter your measurements.")
        else:
            try:
                with st.spinner("Reading your input..."):
                    llm, model, imputer, scaler = load_models()
                    data = extract_features(llm, user_input)

                st.session_state.data = data
                st.rerun()

            except (ValueError, RuntimeError, OSError) as e:
                st.error(str(e))



else:
    data = st.session_state.data

    missing = [
        feature for feature in features
        if data[feature] is None
    ]

    
    if missing and not st.session_state.asked:

        st.write("I'm missing some measurements:")
        st.write(", ".join(missing))

        with st.form("followup"):
            extra_input = st.text_area(
                "Can you provide any of these measurements?"
            )

            submitted = st.form_submit_button("Continue")
            skipped = st.form_submit_button("I don't know")

        if submitted or skipped:
            try:
                if submitted and extra_input.strip():
                    llm, _, _, _ = load_models()
                    extra = extract_features(llm, extra_input)

                    for feature in features:
                        if data[feature] is None:
                            data[feature] = extra[feature]

                st.session_state.asked = True
                st.rerun()

            except (ValueError, RuntimeError) as e:
                st.error(str(e))

    else:
        proceed = False

        if missing:
            st.warning(
                "Still missing: " + ", ".join(missing)
            )

            st.write(
                "I can fill these using the training-data medians, "
                "but the prediction will be less reliable."
            )

            proceed = st.button("Predict using medians")

        else:
            proceed = True

        if proceed:
            try:
                with st.spinner("Running prediction..."):
                    llm, model, imputer, scaler = load_models()

                    prediction, probability = predict_diabetes(
                        data, model, imputer, scaler
                    )

                    explanation = explain_prediction(
                        llm, prediction, probability, missing
                    )

                st.subheader("Model Result")

                if prediction == 1:
                    st.write("**Classification: Diabetes (1)**")
                else:
                    st.write("**Classification: No Diabetes (0)**")

                st.metric(
                    "Positive-class model score",
                    f"{probability:.1%}"
                )

                st.write(explanation)

                if missing:
                    st.info(
                        "Imputed measurements: " + ", ".join(missing)
                    )

            except (ValueError, RuntimeError, OSError) as e:
                st.error(str(e))

    if st.button("Start Over"):
        st.session_state.data = None
        st.session_state.asked = False
        st.rerun()