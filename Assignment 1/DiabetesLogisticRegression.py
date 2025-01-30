"""
  1. Loads the diabetes dataset from sklearn.
  2. Converts the continuous disease-progression target into a binary classification problem
    by labeling each sample as 1 if its progression is above or equal to the median, else 0.
  3. Performs logistic regression of each feature (one at a time) vs. the new binary target.
  4. Plots the resulting data points (jittered in the vertical direction) and the logistic (sigmoid) curves.
  5. Prints the slope, intercept, and classification accuracy for each feature.
  6. Identifies which factor has the best logistic fit by selecting the one with the highest accuracy.
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression

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

# Compute the median of target y
# Make binary array s.t. each element is 1 if above y and 0 if below
# Classification approach
median_target = np.median(y)
# Label: 1 if y >= median, else 0
y_binary = (y >= median_target).astype(int)

# Initialize arrays
slopes = []
intercepts = []
accuracies = []

# Area for plots
plt.figure(figsize=(15, 20))

for i in range(X.shape[1]):
    # Extract the i-th feature as a 2D array
    X_i = X[:, i].reshape(-1, 1)
    
    # Fit a logistic regression model
    log_reg = LogisticRegression(solver='lbfgs')
    log_reg.fit(X_i, y_binary) # Use one feature at a time
    
    # Extract slope (coefficient) and intercept
    slope = log_reg.coef_[0][0]      # log_reg.coef_ is shape (1, 1) for single feature
    intercept = log_reg.intercept_[0]
    
    # Accuracy on training data
    accuracy = log_reg.score(X_i, y_binary)
    
    slopes.append(slope)
    intercepts.append(intercept)
    accuracies.append(accuracy)
    
    # Print out equation and accuracy
    # Logistic function: P(Y=1) = 1 / (1 + exp(-z)), where z = intercept + slope*x
    print(f"Feature {i} ({feature_names[i]}):")
    print(f"  Logistic Equation: P(Y=1) = 1 / (1 + e^(-({slope:.4f} * x + {intercept:.4f})))")
    print(f"  Training Accuracy = {accuracy:.4f}\n")
    
    # Logistic curve, iterate thru features
    plt.subplot(5, 2, i+1)
    
    # Scatter: x-values vs. the binary label (0 or 1), jitter y for visibility
    jittered_y = y_binary + np.random.uniform(-0.05, 0.05, size=len(y_binary))
    plt.scatter(X_i, jittered_y, alpha=0.5, label="Data (jittered in Y)")
    
    # Generate a grid of x-values to draw the logistic curve
    x_min, x_max = X_i.min(), X_i.max()
    x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    # Probability predictions of each class 
    y_prob = log_reg.predict_proba(x_grid)[:, 1]  # Probability of class "1"
    
    # Plot the logistic (sigmoid) curve
    plt.plot(x_grid, y_prob, color="red", label="Logistic curve")
    
    plt.title(f"{feature_names[i]} (Accuracy={accuracy:.2f})")
    plt.xlabel(feature_names[i])
    plt.ylabel("P(Y=1)")
    plt.ylim(-0.1, 1.1)
    plt.legend()

plt.tight_layout()
plt.show()

# identify best feature via highest classification accuracy
best_idx = np.argmax(accuracies)
best_feat_name = feature_names[best_idx]
best_accuracy = accuracies[best_idx]

# Fancy presentation
print("-------------------------------------------------")
print("Which factor has the best logistic fit?")
print(f"The feature '{best_feat_name}' (index {best_idx}) has the highest accuracy = {best_accuracy:.4f}.")
print("-------------------------------------------------")
