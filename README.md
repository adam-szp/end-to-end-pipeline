# End-to-End MLOps Pipeline

## Project Description

This project is an end-to-end machine learning pipeline that predicts diabetes classification using the Pima Indians Diabetes Dataset. The dataset contains 768 entries and 8 numerical features.

The application is designed to demonstrate how machine learning can be combined with a large language model (LLM) to make predictions more accessible to users who may not be familiar with machine learning.

Users can enter their information in natural language through a Streamlit web application. The LLM extracts the relevant features, and the trained classification model makes a prediction on whether or not they have diabetes given their health measurements.

## Setup

Clone the repository:

```powershell
git clone https://github.com/adam-szp/end-to-end-pipeline
cd end-to-end-pipeline
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Download the dataset using DVC:

```powershell
dvc pull
```

DVC is connected to a DagsHub remote. Depending on the remote configuration, DagsHub credentials may be required to download the data.

The application uses a local Hugging Face model, **Qwen2.5-1.5B-Instruct**, so no API key is necessary. The model downloads automatically on its first use. An internet connection is required for the initial download. I used a local model primarily because I could not find an api which offers a free tier. I tried Nebius, HuggingFace, and OpenAI, all of which require credit cards/payment. Secondly I feel that for this experiement a local model provides a good level of editing/customization for the hypothetical company while also being simple enough to run on most computers.

The application also needs access to the saved MLflow model, imputer, and scaler artifacts, all of which are pre-ran and saved in DVC.

## Usage

To run the training pipeline:

```powershell
python -m src.main
```

To run the tests:

```powershell
python -m pytest tests/ -v
```

To launch the Streamlit application:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m streamlit run src/app.py --server.fileWatcherType none
```

Once opened in the browser, users can enter their measurements in natural language.

For example:

"I'm 35, my glucose level is 148, and my BMI is 33.6."

If information is missing, the application asks for additional measurements once. If the user cannot provide them, they can choose to fill the remaining values using the training data's medians.

The application then displays the model's prediction and a short explanation.

## Architecture

The pipeline consists of several steps:

1. **DVC:** Tracks and downloads the dataset.
2. **Preprocessing:** Cleans the data, handles missing values, and scales numerical features.
3. **Training:** Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
4. **Evaluation:** Calculates model performance metrics.
5. **MLflow:** Tracks experiments and stores trained models and preprocessing objects.
6. **LLM Interface:** Qwen extracts measurements from natural language and explains the prediction through Streamlit.

The LLM does not make the diabetes prediction itself. It converts user input into the numerical features required by the best-performing machine learning model, which then gives the prediction back to the LLM to translate into natural language back to the user.

## Results

Several experiments were performed using Logistic Regression, Random Forest, and Gradient Boosting with different hyperparameters.

Recall was chosen as the main evaluation metric because identifying positive diabetes cases is particularly important for this classification problem.

**Best Model:** Logistic Regression: bouncy-eel-47

| Metric | Result |

| Recall | [0.7963] |
| Precision | [0.5972] |
| F1-score | [0.6825] |
| ROC-AUC | [0.8224] |

One important finding was that model complexity does not necessarily lead to better performance, especially with a dataset containing fewer than 800 observations. Another finding was that after decreasing the learning rate on the Gradient Boosting model, the performance got much worse. While technically not more complex this way this was still an interesting find. A basic Logistic Regression was by far the best model. This specific model was chosen since its Recall was the highest of all runs.

## Screen Recording

Please use this link to see the screen recording of the model in action, it will lead to a Google Drive folder: https://drive.google.com/file/d/1diINsXX49eMPI5MEjuNhTZwmJjWBSO_l/view?usp=sharing

## Reflection

This project helped me understand how the different parts of a machine learning pipeline work together, from data preprocessing and model training to automated testing, experiment tracking, and deployment through a simple interface.

Some of the biggest challenges involved setting up DVC with DagsHub, figuring out what kind of LLM/Hosted LLM to use, and connecting the trained model to the LLM interface.

For this particular project, a local LLM is actually a good solution. The task is relatively simple, the application is small, the model can be changed easily, and there are no API costs or usage limits from an external provider. The main tradeoff is that running the LLM requires local computing resources.

With more time, I would experiment with additional models and hyperparameters, improve the LLM's handling of unusual or ambiguous inputs, and test the application on a larger and more diverse dataset.

I would also explore HuggingFace Space's, as that is something new that seems to allow creating a github-like repository while creating a llm of some kind, while also being free for the basic tier. This seems like the perfect solution for this project, however its brand new so I would have to spend some time learning how to connect with Github/DVC or perhaps just using it solely.