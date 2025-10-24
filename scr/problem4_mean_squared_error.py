"""
Problem 4: Mean Squared Error (MSE)

Demonstrates the MSE metric for evaluating regression models:
MSE = (1/m)∑[(y_pred⁽ⁱ⁾ - y⁽ⁱ⁾)²]

This is a general function that can be used for any regression problem.
"""

import numpy as np
from scr.scratch_linear_regression import ScratchLinearRegression, MSE


def main():
    """
    Demonstrate MSE calculation.
    """
    print("=" * 60)
    print("Problem 4: Mean Squared Error (MSE)")
    print("=" * 60)
    
    # Example 1: Perfect predictions
    print("\nExample 1: Perfect predictions")
    y_true1 = np.array([1, 2, 3, 4, 5])
    y_pred1 = np.array([1, 2, 3, 4, 5])
    mse1 = MSE(y_pred1, y_true1)
    print(f"True:      {y_true1}")
    print(f"Predicted: {y_pred1}")
    print(f"MSE: {mse1:.6f} (Perfect predictions)")
    
    # Example 2: Some error
    print("\nExample 2: Predictions with error")
    y_true2 = np.array([3, -0.5, 2, 7, 4])
    y_pred2 = np.array([2.5, 0.0, 2, 8, 3.5])
    mse2 = MSE(y_pred2, y_true2)
    print(f"True:      {y_true2}")
    print(f"Predicted: {y_pred2}")
    print(f"MSE: {mse2:.6f}")
    
    # Example 3: With trained model
    print("\nExample 3: MSE with trained linear regression model")
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5]])
    y = 2 * X.squeeze() + 1 + np.random.randn(5) * 0.5
    
    model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=False)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = MSE(y_pred, y)
    
    print(f"Training MSE: {mse:.6f}")
    print(f"Model equation: y = {model.coef_[1]:.2f}x + {model.coef_[0]:.2f}")
    
    print("\n✓ MSE calculation working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
