''' Approximates e^X where X is a 2x2 matrix '''

import numpy as np
import math

def matrixexp(X, n_terms=10):

    # Initialize result as zero matrix
    expX = np.zeros((2, 2), dtype=float) 

    # Start from identity bc any matrix to the power of zero is I
    current_power = np.eye(2, dtype=float)

    # Sum up to n_terms
    for k in range(n_terms + 1):
        # Add  (1/k!) * (X^k)  to expX
        expX += (1.0 / math.factorial(k)) * current_power
        
        # Update current_power:  X^k -> X^(k+1)
        current_power = current_power @ X
    
    return expX

if __name__ == "__main__":

    X_example = [[1, 2],
                 [3, 5]]
    
    approx_eX = matrixexp(X_example, n_terms=50)
    print(approx_eX)