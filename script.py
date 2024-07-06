# Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Anoma_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Exploratory Data Analysis (EDA) and Data Cleaning
# Data quality check
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Handling missing values (if any)
data.fillna(method='ffill', inplace=True)

# Treat outliers using the IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Convert date column to correct datatype (assuming there is a date column)
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])

# Visualize distributions of features
for column in data.columns:
    if column != 'y' and column != 'date':
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()


# Feature Engineering and Selection
# Example of feature engineering: Creating a new feature
# data['feature_new'] = data['feature1'] * data['feature2'] (modify as per your dataset)

# Feature selection
X = data.drop(['y', 'date'], axis=1) if 'date' in data.columns else data.drop(['y'], axis=1)
y = data['y']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
# Choose and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Model Validation
# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Hyperparameter Tuning
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Best Model Accuracy: {accuracy_best}')
print(classification_report(y_test, y_pred_best))

# Confusion matrix for the best model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Best Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Model Deployment Plan

### Model Deployment Plan
# 1. **Containerization**: Use Docker to create a container for the model.
# 2. **API Creation**: Develop an API using Flask or FastAPI to serve the model.
# 3. **Deployment**: Deploy the container to a cloud platform (AWS, GCP, Azure).
# 4. **Monitoring**: Implement monitoring to track model performance over time.
