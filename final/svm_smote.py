from load_data import load_patient_info, preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import plot_figures
from imblearn.over_sampling import SMOTE

patient_data = load_patient_info()
print(f"Loaded {len(patient_data)} rows, {len(patient_data.columns)} columns from archive/PatientInfo.csv")
print(patient_data.head().to_string(index=False))
# summarize state
print("\nState distribution:")
print(patient_data['state'].value_counts())
X, y = preprocess_features(patient_data)
# SMOTE needs numeric arrays; one-hot columns can be bool, so cast explicitly.
X = X.astype(np.float32)
print(y.value_counts())

# Split the data into training and testing sets
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Initialize SMOTE
smote = SMOTE(random_state=SEED)

# Generate balanced training data
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
print(y_train_smote.value_counts())

# Train a Support Vector Machine model
model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)
model.fit(X_train_smote, y_train_smote)
# Predict on the test set
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
plot_figures.plot_confusion_matrix(y_test, y_pred, save_path='figures/svm/confusion_matrix_smote.png')
plot_figures.plot_precision_recall_curve(y_test, y_scores, save_path='figures/svm/precision_recall_curve_smote.png')
plot_figures.plot_roc_curve(y_test, y_scores, save_path='figures/svm/roc_curve_smote.png')
