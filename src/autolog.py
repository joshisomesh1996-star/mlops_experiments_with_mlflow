import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Define model parameters
max_depth = 10
n_estimators = 5

#set experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops_mlflow_experiment_1")
mlflow.autolog()

#ml-flow
with mlflow.start_run():
    rf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)
    # mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")

    #tags
    mlflow.set_tag("project", "MLOps with MLflow")
    mlflow.set_tag("author", "Somesh Joshi")
    mlflow.set_tag("model", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Wine Quality")

    #log model
    # mlflow.sklearn.log_model(rf, "random_forest_model")

    #Creating confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Save confusion matrix plot
    cm_path = "confusion_matrix.png"
    # plt.savefig(cm_path)
    # mlflow.log_artifact(cm_path)    
    mlflow.log_artifact(__file__)