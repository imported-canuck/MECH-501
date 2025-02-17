### Gradient Descent Example ###
# Wrote this script to understand gradient descent for a midterm #
# Not general/easily mutable, just for demonstration

import numpy as np

# Data points
X = np.array([-4,-2,0,2,4])
Y = np.array([97,-23,0,-23,97])
N = len(X)

# Initialize
a = 0.0
b = 0.0

# Learning rate and iterations
eta = 1e-5 # Increasing causes divergence, don't.
iterations = 100000 # 100000 gives really good convergence, more is overkill 

print("Iteration\t   a\t\t     b\t\t   MSE")
for i in range(iterations):
    # Compute predictions f(x) = a*x^4 + b*x^2 (domain knowledge)
    predictions = a * X**4 + b * X**2
    
    # Compute the errors
    errors = predictions - Y
    
    # Mean Squared Error (MSE)
    mse = np.mean(errors**2)
    
    # Compute gradients
    # Note: For loss function E = (1/N) sum (f(x)-y)^2,
    # the gradient with respect to a is: (2/N)*sum[(f(x)-y)*x^4]
    # and with respect to b is: (2/N)*sum[(f(x)-y)*x^2]
    grad_a = (2.0 / N) * np.sum(errors * X**4)
    grad_b = (2.0 / N) * np.sum(errors * X**2)
    
    # Update the parameters
    a = a - eta * grad_a
    b = b - eta * grad_b
    
    # Print the current iteration, parameters, and error
    print(f"{i+1:>9d}\t {a:>7.4f}\t {b:>7.4f}\t {mse:>10.4f}")

# Final parameters
print("\nFinal parameters:")
print("a =", a)
print("b =", b)
