import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
# Loading and Preprocessing the Dataset
df = pd.read_csv("Housing.csv")

print(df.head()) # Prints first 5 rows of the dataset. 
print(df.info()) # Prints all the information of the dataset.

# Replacing 'yes'/'no' if present
df.replace({'yes': 1, 'no': 0}, inplace=True)

# Checking again
print(df.head())

# For Simple Linear Regression (1 feature: area)

X_simple = df[['area']]
y = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # fixed seed for reproducibility
)

# For Multiple Linear Regression (3 features: area, bedrooms, bathrooms)
X_multiple= df[['area','bedrooms', 'bathrooms']]
y = df['price']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multiple, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # fixed seed for reproducibility
)

# Fit Linear Regression Model

# Initialize models
model_simple = LinearRegression()
model_multiple = LinearRegression()

# Fit models
model_simple.fit(X_train_s, y_train_s)
model_multiple.fit(X_train_m, y_train_m)

# Evaluating Models

# Predictions
y_pred_s = model_simple.predict(X_test_s)
y_pred_m = model_multiple.predict(X_test_m)

# Evaluation Function
def evaluate(y_true, y_pred, model_name="Model"):
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.2f}")

# Evaluate
evaluate(y_test_s, y_pred_s, "Simple Linear Regression")
evaluate(y_test_m, y_pred_m, "Multiple Linear Regression")

# Plot Regression Line (Simple Linear Regression)

plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Predicted')

plt.title('Simple Linear Regression: Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend()
plt.show()

'''We simply cannot plot regression line because 1 feature creates simple 2D line, 2 features creates 
3D plane and 3+ features cannot fully plot but we can create actual VS targeted prices scatter plot.'''

plt.scatter(y_test_m, y_pred_m)
plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], 'k--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()

# Interpret Coefficients

# Simple Linear Regression
print(f"Simple Model Coefficient (slope): {model_simple.coef_[0]:.2f}")
print(f"Simple Model Intercept: {model_simple.intercept_:.2f}")

# Multiple Linear Regression
print("\nMultiple Model Coefficients:")
for feature, coef in zip(X_multiple.columns, model_multiple.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model_multiple.intercept_:.2f}")
