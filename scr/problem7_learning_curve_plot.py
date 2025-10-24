"""
Problem 7: Learning Curve Plot

Create a function to visualize the learning curve showing how the loss
decreases over iterations during training.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scr.scratch_linear_regression import ScratchLinearRegression
import os


def plot_learning_curves(model, save_path="plots/problem7_learning_curve.png"):
    """
    Plot training and validation learning curves.
    
    Parameters
    ----------
    model : ScratchLinearRegression
        Trained model with recorded loss history
    save_path : str
        Path where the plot will be saved
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    iterations = range(1, len(model.loss) + 1)
    plt.plot(iterations, model.loss, label='Training Loss', 
             color='blue', linewidth=2, alpha=0.8)
    
    # Plot validation loss if available
    if np.any(model.val_loss):
        plt.plot(iterations, model.val_loss, label='Validation Loss',
                color='red', linewidth=2, linestyle='--', alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss J(θ)', fontsize=12)
    plt.title('Learning Curves: Training and Validation Loss vs Iterations', 
              fontsize=14, pad=20)
    
    # Use log scale for better visualization
    plt.yscale('log')
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Demonstrate learning curve plotting.
    """
    print("=" * 60)
    print("Problem 7: Learning Curve Plot")
    print("=" * 60)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2.5 * X.squeeze() + 3.0 + np.random.randn(100) * 2.0
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train model
    print("\nTraining model...")
    model = ScratchLinearRegression(num_iter=500, lr=0.01, no_bias=False, verbose=False)
    model.fit(X_train, y_train, X_val, y_val)
    
    print(f"\nTraining complete!")
    print(f"Initial training loss: {model.loss[0]:.6f}")
    print(f"Final training loss: {model.loss[-1]:.6f}")
    print(f"Initial validation loss: {model.val_loss[0]:.6f}")
    print(f"Final validation loss: {model.val_loss[-1]:.6f}")
    print(f"Loss reduction: {((model.loss[0] - model.loss[-1]) / model.loss[0] * 100):.2f}%")
    
    # Plot and save learning curves
    os.makedirs('plots', exist_ok=True)
    plot_learning_curves(model)
    
    print("\n✓ Learning curve plot saved to plots/problem7_learning_curve.png")
    print("  The plot shows loss decreasing over iterations, indicating successful learning.")
    print("=" * 60)


if __name__ == "__main__":
    main()
