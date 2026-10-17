# Employee Attrition MLOps Pipeline
## Project Overview
This project builds an end-to-end MLOps pipeline for predicting employee attrition using a csv employee dataset.

The focus of the project was not an advanced model development, but the pipeline itself. The project includes data versioning with Git and DVC, experiment tracking with MLflow, automated testing with pytest, CI/CD with GitHub Actions, and data drift monitoring with Evidently.

The target variable represents employee employment status and is encoded as:
- Active = 0
- Terminated = 1
The model is trained as a classification problem to predict whether an employee is likely to be terminated. 

## Setup
### Disclaimer: According to the project requirements DVC should not be versioned/tracked on Git, however, at the same time it is required to have a remote configured for DVC where the files are stored, and a local remote is permitted. With that in mind, and in order for the project to be easily reproducable, I set up the local remote in a seperate folder/directory, while at the same time properly adding the standard paths to .gitignore. dvc_store/ holds the actual data, and it is logged in Git.
Clone the repository:
```powershell
git clone https://github.com/adam-szp/mlops-pipeline
cd mlops-pipeline
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

Pull the dataset with DVC:
```powershell
dvc pull
```

## Run Training
Run the training pipeline with:
```powershell
python src/train.py
```

The script loads the configuration file, preprocesses the data, trains the model, and lods the run to MLflow.
### You can adjust the parameters in the config file located withing configs/

## Run Tests
Run the full pytest suite with:
```powershell
pytest tests/ -v
```

The test suite includes preprocessing tests, data validation tests, and model validation tests.

## Drift Monitoring
Run drift monitoring with:
```powershell
python src/monitor_drift.py
```

The script compares the training dataset against simulated production data using Evidently and saves an HTML report in the reports/ directory.

## Drift Analysis
### Which features showed drift and why?
The features that showed drift were: age, length_of_service, and birthdate_key.
These features drifted because the simulated production dataset intentionally changed their distributions compared with the reference training data. birthdate_key is included however this "feature", among others, are dropped during actual training since it/they do not provide any substantial information to the model. (age is its own feature)

### Would this drift likely affect model performance?
Possibly. If the drifted features are important to the model's predictions, changes in their distributions could reduce model performance on new data. Over time this could result in a large drift from true predictions. However in this pipeline, feature importance has not been checked.

### What action would you recommend?
If the drifting features were important and truely drifting, not simply simulated, then I would recommend retraining. This would signify a larger problem down the line, as compounding errors would pile up into a catastrophic failure.
If the features were unimportant, but still drifting heavily, I would recommend investigating the reason behind the shifts. There could be an underlying trend leading to a different, perhaps actually important, problem/feature.
If the features were both unimportant and not shifting heavily, then I would recommend just observing, as the impact is minimal and going in to figure it out as well as update the pipeline for no real reason is just wasting time.