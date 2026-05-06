import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('breast-cancer.csv')

# Display first few rows of the dataset
print(df.head())

# Display dataset shape
print("Dataset shape:", df.shape)

# Display dataset summary
print(df.describe())

# Check for missing values and handle them
print("Missing values:\n", df.isnull().sum())
df_cleaned = df.dropna()
print("Missing values after cleaning:\n", df_cleaned.isnull().sum())

# Separate features (X) and target (y)
x = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']
# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the feature data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
joblib.dump(scaler, 'scaler.pkl')

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Define the SVM model
svm_model = SVC()

# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto'],
#     'degree': [3, 4, 5]
# }

# # Perform GridSearchCV
# grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
# grid_search.fit(x_train, y_train)

# # Display the best parameters and use the best model
# print("Best parameters found by GridSearchCV:", grid_search.best_params_)
# best_svm_model = grid_search.best_estimator_

# # Make predictions
# y_pred = best_svm_model.predict(x_test)

# # Evaluate the model
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# # Plot ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, best_svm_model.decision_function(x_test))
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc='lower right')
# plt.show()

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 4, 5]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(svm_model, param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)

# Display the best parameters and use the best model
print("Best parameters found by RandomizedSearchCV:", random_search.best_params_)
best_svm_model_random = random_search.best_estimator_

# Make predictions
y_pred_random = best_svm_model_random.predict(x_test)

# Evaluate the model
print("Classification Report (RandomizedSearchCV):\n", classification_report(y_test, y_pred_random))
print("Confusion Matrix (RandomizedSearchCV):\n", confusion_matrix(y_test, y_pred_random))
print("Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_random))
print("ROC AUC Score (RandomizedSearchCV):", roc_auc_score(y_test, y_pred_random))

# Plot ROC Curve for RandomizedSearchCV
fpr_random, tpr_random, thresholds_random = roc_curve(y_test, best_svm_model_random.decision_function(x_test))
plt.figure()
plt.plot(fpr_random, tpr_random, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_random))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - RandomizedSearchCV')
plt.legend(loc='lower right')
plt.show()
joblib.dump(best_svm_model_random, 'svm_model.pkl')
print("SVM model saved as 'svm_model.pkl'")



#-----------------------------------Random Forest Classifier-----------------------------------
print('-----------------------------------Random Forest Classifier-----------------------------------')



# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# # Perform GridSearchCV
# grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)
# grid_search.fit(x_train, y_train)

# # Display the best parameters and use the best model
# print("Best parameters found by GridSearchCV:", grid_search.best_params_)
# best_rf_model = grid_search.best_estimator_

# # Make predictions
# y_pred = best_rf_model.predict(x_test)

# # Evaluate the model
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# # Plot ROC Curve
# y_pred_prob = best_rf_model.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_prob))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc='lower right')
# plt.show()

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(rf_model, param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)

# Display the best parameters and use the best model
print("Best parameters found by RandomizedSearchCV:", random_search.best_params_)
best_rf_model_random = random_search.best_estimator_

# Make predictions
y_pred_random = best_rf_model_random.predict(x_test)

# Evaluate the model
print("Classification Report (RandomizedSearchCV):\n", classification_report(y_test, y_pred_random))
print("Confusion Matrix (RandomizedSearchCV):\n", confusion_matrix(y_test, y_pred_random))
print("Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_random))
print("ROC AUC Score (RandomizedSearchCV):", roc_auc_score(y_test, y_pred_random))

# Plot ROC Curve for RandomizedSearchCV
y_pred_prob_random = best_rf_model_random.predict_proba(x_test)[:, 1]
fpr_random, tpr_random, thresholds_random = roc_curve(y_test, y_pred_prob_random)
plt.figure()
plt.plot(fpr_random, tpr_random, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_prob_random))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - RandomizedSearchCV')
plt.legend(loc='lower right')
plt.show()

joblib.dump(best_rf_model_random, 'random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'")
results = pd.DataFrame({
    "Model": ["SVM", "Random Forest"],
    "Accuracy": [0.982, 0.956],
    "ROC AUC": [roc_auc_score(y_test, y_pred_random), roc_auc_score(y_test, y_pred_prob_random)]
})
print(results)