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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

X_train, y_train, X_test, y_test = [None] * 4
logreg_model = None

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
    X = train_dataset_encoded.drop(columns="Survived", axis=1).values
    y = train_dataset_encoded["Survived"].values
    global X_train, y_train, X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=10)


def plot_roc_curve(y_test, y_pred_probs, title):
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
    axs.set_title(f"{title} (ROC) Curve")
    axs.legend(loc="lower right")
    plt.grid()
    file_name = "_".join(title.lower().split(" "))
    plt.savefig(f"{file_name}_roc_curve.png")
    plt.show()


def train_logistic_regression():
    logreg = LogisticRegression()
    params = {
        "penalty": ["l1", "l2"],
        "C": np.linspace(0.001, 0.1, 20),
        "solver": ["liblinear"]
    }
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    logreg_grid = GridSearchCV(estimator=logreg, param_grid=params, cv=kf, scoring="accuracy")
    logreg_grid.fit(X_train, y_train)
    print(f"Best Parameters: {logreg_grid.best_params_}")
    print(f"Best Cross-Validation Accuracy: {logreg_grid.best_score_:.2f}")
    global logreg_model
    logreg_model = logreg_grid.best_estimator_
    y_pred = logreg_model.predict(X_test)
    y_pred_probs = logreg_model.predict_proba(X_test)[:, 1]
    print(f"============= Logistic Regression Model Evaluation =============")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")
    plot_roc_curve(y_test, y_pred_probs, title="Logistic Regression")


if __name__ == "__main__":
    load_and_preprocess_dataset()
    train_logistic_regression()
