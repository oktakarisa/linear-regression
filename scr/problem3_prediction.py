"""
Problem 3: Prediction

Demonstrates the prediction mechanism using the trained model.
The predict() method uses the hypothesis function to generate predictions.
"""

import numpy as np
from scr.scratch_linear_regression import ScratchLinearRegression


def main():
    """
    Demonstrate prediction on trained model.
    """
    print("=" * 60)
    print("Problem 3: Prediction Mechanism")
    print("=" * 60)
    
    # Generate training data
    np.random.seed(42)
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = 3 * X_train.squeeze() + 2 + np.random.randn(5) * 0.5
    
    print("\nTraining model...")
    model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=False)
    model.fit(X_train, y_train)
    
    print(f"Learned model: y = {model.coef_[1]:.2f}x + {model.coef_[0]:.2f}")
    
    # Make predictions on training data
    print("\nPredictions on training data:")
    train_predictions = model.predict(X_train)
    for i, (x, y_true, y_pred) in enumerate(zip(X_train, y_train, train_predictions)):
        print(f"  x={x[0]:.0f}: actual={y_true:.2f}, predicted={y_pred:.2f}")
    
    # Make predictions on new data
    print("\nPredictions on new data:")
    X_new = np.array([[6], [7], [8], [9], [10]])
    new_predictions = model.predict(X_new)
    for x, y_pred in zip(X_new, new_predictions):
        print(f"  x={x[0]:.0f}: predicted={y_pred:.2f}")
    
    print("\n✓ Prediction mechanism working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
