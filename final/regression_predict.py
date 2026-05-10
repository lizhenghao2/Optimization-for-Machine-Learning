from load_data import load_patient_info, preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import plot_figures

patient_data = load_patient_info()
print(f"Loaded {len(patient_data)} rows, {len(patient_data.columns)} columns from archive/PatientInfo.csv")
print(patient_data.head().to_string(index=False))
# summarize state
print("\nState distribution:")
print(patient_data['state'].value_counts())
X, y = preprocess_features(patient_data)
print(y.value_counts())

# Split the data into training and testing sets
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
# Train a logistic regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:,1]

threshold = 0.1

y_pred = (
    y_scores > threshold
).astype(int)

print(classification_report(y_test, y_pred))

plot_figures.plot_confusion_matrix(y_test, y_pred, save_path='figures/logistic_regression/confusion_matrix.png')
plot_figures.plot_precision_recall_curve(y_test, y_scores, save_path='figures/logistic_regression/precision_recall_curve.png')
plot_figures.plot_roc_curve(y_test, y_scores, save_path='figures/logistic_regression/roc_curve.png')
plot_figures.plot_feature_importance(model, X.columns, save_path='figures/logistic_regression/feature_importance.png')