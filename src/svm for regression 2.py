import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of displaying minus sign

# Read CSV file
data = pd.read_csv('the true one.csv')

# Print column names
print(data.columns)

# Data exploration
print(data.head())
print(data.info())
print(data.describe())

# Data visualization
sns.pairplot(data)
plt.show()

# Check for missing values
missing_values = data.isnull().sum()
print("Number of missing values in each column:")
print(missing_values)

# Fill missing values with column mean
data = data.fillna(data.mean())

# Separate features and target variable
X = data.drop('Target (Total orders)', axis=1)
y = data['Target (Total orders)']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVR model
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

# Define K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use grid search for hyperparameter tuning
param_grid = {
    'svr__C': [0.1, 1, 10, 100],  # Modify parameter names
    'svr__epsilon': [0.01, 0.1, 0.2, 0.5],  # Modify parameter names
    'svr__gamma': ['scale', 'auto']  # Modify parameter names
}
grid_search = GridSearchCV(svr, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print(f'Best parameters: {grid_search.best_params_}')

# Train model with best parameters
best_svr = grid_search.best_estimator_

# Evaluate model with cross-validation
mse_scores = cross_val_score(best_svr, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
mean_mse = -mse_scores.mean()
print(f'Average MSE: {mean_mse}')

# Predict on test set and calculate MSE and R²
best_svr.fit(X_train, y_train)
y_pred = best_svr.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f'Test set MSE: {test_mse}')
print(f'Test set R²: {test_r2}')

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()
