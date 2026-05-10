from load_data import load_patient_info, preprocess_features
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
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

# Grid search for the best SVM configuration
param_grid = {
	'kernel': ['linear', 'rbf'],
	'C': [0.01, 0.1, 1, 10, 100],
	'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(
	SVC(probability=True, class_weight='balanced', random_state=SEED),
	param_grid=param_grid,
	scoring='f1',
	cv=cv,
	n_jobs=-1,
	verbose=1,
)
grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the best model on the training set
model = grid_search.best_estimator_

# Predict on the test set
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
plot_figures.plot_confusion_matrix(y_test, y_pred, save_path='figures/svm/confusion_matrix_grid.png')
plot_figures.plot_precision_recall_curve(y_test, y_scores, save_path='figures/svm/precision_recall_curve_grid.png')
plot_figures.plot_roc_curve(y_test, y_scores, save_path='figures/svm/roc_curve_grid.png')
