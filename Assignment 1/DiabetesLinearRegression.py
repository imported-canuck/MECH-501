"""
  1. Loads the diabetes dataset from sklearn.
  2. Performs linear regression of each feature (one at a time) vs. the target.
  3. Plots the resulting scatter plots and regression lines.
  4. Prints the slope, intercept, and R^2 for each feature.
  5. Identifies which factor seems most linearly related to the target by maximizing R^2. 
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Load the data
diabetes = load_diabetes()
X = diabetes.data  # shape: (442, 10)
y = diabetes.target  # shape: (442,)
feature_names = [
    "age",  # Param 0
    "sex",  # " 1
    "bmi",  # " 2
    "bp",   # " 3
    "s1",   # " 4
    "s2",   # " 5
    "s3",   # " 6
    "s4",   # " 7
    "s5",   # " 8
    "s6"    # " 9
]

# Set up arrays for regressions
slopes = []
intercepts = []
r2_scores = []

# Main plot
plt.figure(figsize=(15, 20))

for i in range(X.shape[1]): # Loop through each feature 
    X_i = X[:, i].reshape(-1, 1) # Reshape changes it into 442 x 1 array for regression
    
    # Fit a linear regression model
    reg = LinearRegression()
    reg.fit(X_i, y)
    
    # Store slope, intercept, R^2
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(X_i, y)
    
    # Track slope, intercept, R^2
    slopes.append(slope)
    intercepts.append(intercept)
    r2_scores.append(r2)
    
    # Print the linear equation and R^2
    # Equation of line: y = slope * x + intercept
    print(f"Feature {i} ({feature_names[i]}):")
    print(f"    Equation: y = {slope:.4f} * x + {intercept:.4f}")
    print(f"    R^2 = {r2:.4f}\n")
    
    # Plots
    plt.subplot(5, 2, i+1)  # 5 rows, 2 columns, i-th subplot (bc we have 10 params)
    plt.scatter(X_i, y, alpha=0.5, label="Data")
    
    # Create a sequence of x-values to plot the regression line
    x_line = np.linspace(X_i.min(), X_i.max(), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)
    plt.plot(x_line, y_line, color="red", label="Best fit line")
    
    plt.title(f"Feature {i}: {feature_names[i]}")
    plt.xlabel(feature_names[i])
    plt.ylabel("Disease Progression")
    plt.legend()

plt.tight_layout()
plt.show()

# Closest to linear is the param with the highest R^2 value
best_idx = np.argmax(r2_scores)
best_feature_name = feature_names[best_idx]
best_r2 = r2_scores[best_idx]

# Fancy formatting
print("-------------------------------------------------")
print("Which factor is most linearly related?")
print(f"Based on single-factor R^2 values, the feature '{best_feature_name}' "
      f"(index {best_idx}) shows the highest R^2 of {best_r2:.4f}.")
print("Hence, it appears to have the strongest linear relationship with disease progression.")
print("-------------------------------------------------")
