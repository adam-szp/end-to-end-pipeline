from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Parameters:
    model: The trained model.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The true labels for the test set.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_pred, average='weighted')
    }
    
    print(f"Model Evaluation:\nAccuracy: {round(metrics['accuracy'], 4)}\nPrecision: {round(metrics['precision'], 4)}\nRecall: {round(metrics['recall'], 4)}\nF1 Score: {round(metrics['f1_score'], 4)}\nAUC ROC: {round(metrics['roc_auc'], 4)}")

    return metrics