"""
Author: Sai Sundeep Rayidi
Date: 7/17/2024

Description:
[Description of what the file does, its purpose, etc.]

Additional Notes:
[Any additional notes or information you want to include.]

License: 
MIT License

Copyright (c) 2024 Sai Sundeep Rayidi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contact:
[Optional: How to reach you for questions or collaboration.]

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

X_train, y_train, X_test, y_test = [None] * 4
logreg_model, knn_model, dt_clf = [None] * 3
training_features = []


def load_and_preprocess_dataset():
    train_dataset = pd.read_csv("train.csv")
    test_dataset = pd.read_csv(f"test.csv")
    train_dataset_encoded = pd.get_dummies(data=train_dataset, columns=["Sex", "Pclass", "Embarked"], dtype=int)
    test_dataset_encoded = pd.get_dummies(data=test_dataset, columns=["Sex", "Pclass", "Embarked"], dtype=int)
    train_dataset_encoded.drop(columns=["PassengerId", "Ticket", "Name", "Cabin"], axis=1, inplace=True)
    test_dataset_encoded.drop(columns=["PassengerId", "Ticket", "Name", "Cabin"], axis=1, inplace=True)
    imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
    train_dataset_encoded["Age"] = imputer.fit_transform(train_dataset_encoded[["Age"]])
    train_dataset_encoded["Age"] = imputer.fit_transform(train_dataset_encoded[["Age"]])
    global X_train, y_train, X_test, y_test, training_features
    X = train_dataset_encoded.drop(columns="Survived", axis=1)
    training_features = X.columns
    X = X.values
    y = train_dataset_encoded["Survived"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=10, stratify=y)


def plot_roc_curve(y_test, y_pred_probs, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    auc = roc_auc_score(y_test, y_pred_probs)
    print(f"ROC AUC Score: {auc:.2f}")
    fig, axs = plt.subplots(figsize=(8, 6))
    axs.plot(fpr, tpr, color="blue", lw=3, label="ROC_CURVE")
    axs.plot([0, 1], [0, 1], color="black", linestyle="--")
    axs.fill_between(fpr, tpr, color="blue", alpha=0.3)
    axs.text(s=f"AUC Score: {auc:.2f}", x=0.7, y=0.3, fontweight="bold")
    axs.set_xlim([0.0, 1.0])
    axs.set_ylim([0.0, 1.05])
    axs.set_xlabel('False Positive Rate (1 - Specificity)')
    axs.set_ylabel('True Positive Rate (Sensitivity)')
    axs.set_title(f"{model_name} (ROC) Curve")
    axs.legend(loc="lower right")
    plt.grid()
    file_name = "_".join(model_name.lower().split(" "))
    plt.savefig(f"{file_name}_roc_curve.png")
    plt.show()


def plot_feature_importance(estimator_coefs, model_name):
    plt.figure(figsize=(10, 6))
    plt.barh(training_features, estimator_coefs, color="skyblue")
    plt.xlabel(f"Coefficient Magnitude")
    plt.ylabel(f"Feature")
    plt.title(f"Feature Importance")
    file_name = "_".join(model_name.lower().split(" "))
    plt.savefig(f"{file_name}_feature_importance_plot.png")
    plt.show()


def plot_precision_recall_curve(precision, recall, average_precision, model_name):
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color="darkblue", where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.3, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.title(f"Precision-Recall Curve: AP:{average_precision:.2f}")
    file_name = "_".join(model_name.lower().split(" "))
    plt.savefig(f"{file_name}_precision_recall_curve.png")
    plt.show()


def generate_classification_metrics(estimator, model_name, y_pred, y_pred_probs, X_test_scaled):
    print(f"============= {model_name} Model Evaluation =============")
    file_name = "_".join(model_name.lower().split(" "))
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
    _ = ConfusionMatrixDisplay.from_estimator(estimator, X_test_scaled, y_test)
    plt.savefig(f"{file_name}_classifier_confusion_matrix.png")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")
    print(f"{model_name} Precision Score: {precision_score(y_test, y_pred):.2f}")
    print(f"{model_name} Recall Score {recall_score(y_test, y_pred):.2f}")
    print(f"{model_name} Accuracy Score {accuracy_score(y_test, y_pred):.2f}")
    print(f"{model_name} F1 Score {f1_score(y_test, y_pred):.2f}")

    # Generate Plots
    plot_roc_curve(y_test, y_pred_probs, model_name=model_name)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    average_precision = average_precision_score(y_test, y_pred_probs)
    plot_precision_recall_curve(precision=precision,
                                recall=recall,
                                average_precision=average_precision,
                                model_name=model_name)
    print("=" * 80)


def train_logistic_regression():
    steps = [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression())
    ]

    params = {
        "logreg__penalty": ["l1", "l2"],
        "logreg__C": np.linspace(0.001, 0.1, 20),
        "logreg__solver": ["liblinear"]
    }

    pipeline = Pipeline(steps)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    logreg_grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=kf, scoring="accuracy")
    logreg_grid.fit(X_train, y_train)

    print(f"Best Parameters: {logreg_grid.best_params_}")
    print(f"Best Cross-Validation Accuracy: {logreg_grid.best_score_:.2f}")

    global logreg_model
    logreg_model = logreg_grid.best_estimator_
    X_train_scaled = logreg_model.named_steps["scaler"].transform(X_train)
    X_test_scaled = logreg_model.named_steps["scaler"].transform(X_test)
    y_pred = logreg_model.predict(X_test_scaled)
    y_pred_probs = logreg_model.predict_proba(X_test_scaled)[:, 1]

    generate_classification_metrics(estimator=logreg_model, model_name="Logistic Regression",
                                    y_pred=y_pred, y_pred_probs=y_pred_probs, X_test_scaled=X_test_scaled)


def train_knn_classifier():
    steps = [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ]
    params = {
        "knn__n_neighbors": list(np.arange(3, 16)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["manhattan", "euclidean"]
    }

    pipeline = Pipeline(steps)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    knn_grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=kf, scoring="accuracy")
    knn_grid.fit(X_train, y_train)
    print(f"Best Parameters: {knn_grid.best_params_}")
    print(f"Best Cross-Validation Accuracy: {knn_grid.best_score_:.2f}")

    global knn_model
    knn_model = knn_grid.best_estimator_
    X_train_scaled = knn_model.named_steps['scaler'].transform(X_train)
    X_test_scaled = knn_model.named_steps['scaler'].transform(X_test)
    y_pred = knn_model.predict(X_test_scaled)
    y_pred_probs = knn_model.predict_proba(X_test_scaled)[:, 1]

    generate_classification_metrics(estimator=knn_model, model_name="KNeighborsClassifier",
                                    y_pred=y_pred, y_pred_probs=y_pred_probs, X_test_scaled=X_test_scaled)


def unconstrained_dt_classifier():
    dt_clf = DecisionTreeClassifier(random_state=10)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)

    print(f"============= Decision Tree Classifier Evaluation (Un-constrained) =============")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
    _ = ConfusionMatrixDisplay.from_estimator(dt_clf, X_test, y_test)
    plt.savefig(f"decision_tree_classifier_confusion_matrix.png")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")
    print(f"Decision Tree Precision Score: {precision_score(y_test, y_pred):.2f}")
    print(f"Decision Tree Recall Score {recall_score(y_test, y_pred):.2f}")
    print(f"Decision Tree Accuracy Score {accuracy_score(y_test, y_pred):.2f}")
    print(f"Decision Tree F1 Score {f1_score(y_test, y_pred):.2f}")
    print(f"=" * 80)


def train_decsion_tree_classifier():
    unconstrained_dt_classifier()
    params = {
        "criterion": ["gini", "log_loss", "entropy"],
        "max_features": [5, 7, 10, "log2", "sqrt", None],
        "max_depth": [3, 4, 5, 6, 7, None],
        "min_samples_leaf": np.linspace(0.001, 0.0001, 10)
    }
    kf = KFold(n_splits=7, shuffle=True, random_state=10)
    dt_clf_grid = GridSearchCV(estimator=DecisionTreeClassifier(random_state=10),
                                     param_grid=params,
                                     cv=kf, scoring="accuracy",
                                     # random_state=10
                               )
    dt_clf_grid.fit(X_train, y_train)
    print(f"Best Parameters: {dt_clf_grid.best_params_}")
    print(f"Best Cross-Validation Accuracy: {dt_clf_grid.best_score_:.2f}")

    global dt_clf
    dt_clf = dt_clf_grid.best_estimator_
    y_pred = dt_clf.predict(X_test)
    y_pred_probs = dt_clf.predict_proba(X_test)[:, 1]

    generate_classification_metrics(estimator=dt_clf, model_name="Decision Tree",
                                    y_pred=y_pred, y_pred_probs=y_pred_probs, X_test_scaled=X_test)


def train_dt_bagging_classifier():
    dt_clf = DecisionTreeClassifier(min_samples_leaf=1, max_depth=8, random_state=10)
    bc = BaggingClassifier(estimator=dt_clf, n_estimators=300, n_jobs=-1)
    bc.fit(X_train, y_train)
    y_pred = bc.predict(X_test)
    y_pred_probs = bc.predict_proba(X_test)[:, 1]

    generate_classification_metrics(estimator=bc, model_name="Bagging With Decision Tree",
                                    y_pred=y_pred, y_pred_probs=y_pred_probs, X_test_scaled=X_test)


if __name__ == "__main__":
    load_and_preprocess_dataset()
    train_logistic_regression()
    train_knn_classifier()
    train_decsion_tree_classifier()
    train_dt_bagging_classifier()
