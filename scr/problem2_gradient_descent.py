"""
Problem 2: Gradient Descent

Demonstrates the gradient descent optimization algorithm:
θⱼ := θⱼ - α(1/m)∑[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾]

where:
- α is the learning rate
- m is the number of samples
"""

import numpy as np
from scr.scratch_linear_regression import ScratchLinearRegression


def main():
    """
    Demonstrate gradient descent training on a simple linear dataset.
    """
    print("=" * 60)
    print("Problem 2: Gradient Descent")
    print("=" * 60)
    
    # Generate simple linear data: y = 2x + 1
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5]])
    y = 2 * X.squeeze() + 1 + np.random.randn(5) * 0.1
    
    print("\nTraining data:")
    print("X:", X.squeeze())
    print("y:", y)
    
    # Train model
    print("\nTraining with gradient descent...")
    model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=True)
    model.fit(X, y)
    
    print(f"\nLearned coefficients:")
    print(f"  θ₀ (bias) = {model.coef_[0]:.4f}")
    print(f"  θ₁ (weight) = {model.coef_[1]:.4f}")
    
    print(f"\nExpected coefficients (from y = 2x + 1):")
    print(f"  θ₀ (bias) ≈ 1.0")
    print(f"  θ₁ (weight) ≈ 2.0")
    
    # Make predictions
    predictions = model.predict(X)
    print(f"\nPredictions vs Actual:")
    for i, (x_val, pred, actual) in enumerate(zip(X, predictions, y)):
        print(f"  x={x_val[0]:.0f}: predicted={pred:.4f}, actual={actual:.4f}, error={abs(pred-actual):.4f}")
    
    print("\n✓ Gradient descent training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
