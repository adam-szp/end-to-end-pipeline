import math
import pandas as pd
import streamlit as st
from transformers import pipeline
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

from src.compare_experiments import load_best_artifacts


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
        model="Qwen/Qwen2.5-1.5B-Instruct",
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
        max_new_tokens=350,
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

    Recognize natural language and common alternative names
    for each feature (e.g., age, blood sugar, blood pressure,
    pregnancy count, skinfold thickness, insulin, BMI, family history score).

    Return JSON with all features and an "in_scope" boolean.
    Use numbers for known values and null for missing or ambiguous values.
    Never guess missing measurements or calculate unknown scores.
    Set in_scope to false only for unrelated requests.
    Return JSON only.
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
    Explain the diabetes model's prediction in a natural,
    conversational way that is easy for anyone to understand.

    Explain what the result means without repeating technical
    terms, raw model outputs, or unnecessary details.
    Keep it brief, friendly, and direct, but do not say something like "the model thinks you have diabetes".

    If measurements were estimated, briefly mention that they were estimated.
    Don't diagnose the user or present the result as medical certainty.
    Don't repeat the exact percentage because it is already displayed.
    """

    user_input = f"""
    Prediction: {prediction} (1 = diabetes, 0 = no diabetes)
    Model score for diabetes: {probability:.4f}
    Estimated measurements: {missing}
    """

    return ask_llm(llm, prompt, user_input)

def main():

    st.title("Diabetes Classification Assistant")

    st.write(
        "Describe your measurements and I'll use a trained "
        "machine learning model to make a classification."
        "You can include any of the following features:"
    )

    with st.expander("View supported measurements"):
        st.markdown("""
        - **Pregnancies** — Number of times pregnant
        - **Glucose** — Blood glucose level (mg/dL)
        - **BloodPressure** — Diastolic blood pressure (mmHg)
        - **SkinThickness** — Triceps skinfold thickness (mm)
        - **Insulin** — 2-hour serum insulin (µU/mL)
        - **BMI** — Body Mass Index
        - **DiabetesPedigreeFunction** — Diabetes pedigree function score
        - **Age** — Age in years
        """)

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
                        st.write("**Classification: Diabetes**")
                    else:
                        st.write("**Classification: No Diabetes**")

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

if __name__ == '__main__':
    main()