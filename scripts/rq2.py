import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from fraudDetection import CreditCardFraudDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score

dataset = CreditCardFraudDataset(file_path="dataset/creditcard_2023.csv", train=True)
X = dataset.features
y = dataset.labels
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#models
model_logistic_regression = LogisticRegression()
model_decision_tree = DecisionTreeClassifier()
model_random_forest = RandomForestClassifier()
model_svm = SVC(probability=True)

####
model = model_svm # change model here
####

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auprc = average_precision_score(y_test, y_scores)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUPRC: {auprc:.4f}")


feature_importances = None

if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
    feature_importances = model.feature_importances_
elif isinstance(model, LogisticRegression):
    feature_importances = abs(model.coef_[0])
elif isinstance(model, SVC) and model.kernel == 'linear':
    feature_importances = abs(model.coef_[0])

if feature_importances is not None:
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)
    print("\nTop 10 most relevant features:")
    print(importance_df.head(10))