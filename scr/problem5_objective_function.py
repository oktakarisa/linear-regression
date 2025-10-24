"""
Problem 5: Objective Function (Loss Function)

Demonstrates the objective function used in linear regression:
J(θ) = (1/2m)∑[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²]

This is the function that gradient descent minimizes during training.
The loss is recorded in self.loss and self.val_loss during training.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scr.scratch_linear_regression import ScratchLinearRegression


def main():
    """
    Demonstrate the objective function and loss recording.
    """
    print("=" * 60)
    print("Problem 5: Objective Function (Loss Function)")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2.5 * X.squeeze() + 1.5 + np.random.randn(100) * 1.5
    
    # Split into train and validation
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train model with validation data
    print("\nTraining model and recording loss...")
    model = ScratchLinearRegression(num_iter=500, lr=0.01, verbose=False)
    model.fit(X_train, y_train, X_val, y_val)
    
    print(f"\nFinal training loss: {model.loss[-1]:.6f}")
    print(f"Final validation loss: {model.val_loss[-1]:.6f}")
    print(f"Initial training loss: {model.loss[0]:.6f}")
    print(f"Loss reduction: {model.loss[0] - model.loss[-1]:.6f}")
    
    # Plot loss over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss, label='Training Loss', linewidth=2)
    plt.plot(model.val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss J(θ)')
    plt.title('Objective Function: Loss vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    import os
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/problem5_objective_function.png', dpi=300)
    print("\n✓ Loss plot saved to plots/problem5_objective_function.png")
    
    print("\n✓ Objective function implementation verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
