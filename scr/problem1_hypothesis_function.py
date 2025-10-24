"""
Problem 1: Hypothesis Function

Demonstrates the implementation of the linear hypothesis function:
hθ(x) = θ₀x₀ + θ₁x₁ + ... + θₙxₙ (where x₀ = 1)

In vector form: hθ(x) = θᵀx
"""

import numpy as np
from scr.scratch_linear_regression import ScratchLinearRegression


def main():
    """
    Demonstrate the hypothesis function with a simple example.
    """
    print("=" * 60)
    print("Problem 1: Hypothesis Function")
    print("=" * 60)
    
    # Create model instance
    model = ScratchLinearRegression(no_bias=False)
    
    # Set coefficients manually for demonstration
    # bias = 2.0, weight = 3.0
    model.coef_ = np.array([2.0, 3.0])
    
    # Test data
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    
    print("\nModel coefficients (θ):")
    print(f"  θ₀ (bias) = {model.coef_[0]}")
    print(f"  θ₁ (weight) = {model.coef_[1]}")
    
    print("\nInput features (X):")
    print(X)
    
    # Make predictions using hypothesis function
    predictions = model.predict(X)
    
    print("\nPredictions using hypothesis function hθ(x) = θ₀ + θ₁x₁:")
    for i, (x_val, pred) in enumerate(zip(X, predictions)):
        print(f"  x = {x_val[0]:.1f}  →  hθ(x) = {model.coef_[0]} + {model.coef_[1]} × {x_val[0]:.1f} = {pred:.1f}")
    
    print("\n✓ Hypothesis function working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
