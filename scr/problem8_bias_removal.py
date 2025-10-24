"""
Problem 8: Bias Term Removal (Advanced)

Investigate how learning changes when the bias term θ₀ is removed.
Explores the role of the bias term in linear regression models.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scr.scratch_linear_regression import ScratchLinearRegression, MSE
import os


def main():
    """
    Compare models with and without bias term.
    """
    print("=" * 60)
    print("Problem 8: Bias Term Removal (Advanced)")
    print("=" * 60)
    
    # Generate data that does NOT pass through origin
    # This will show the importance of bias term
    print("\nGenerating data with non-zero intercept (y = 3x + 5)...")
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 3 * X.squeeze() + 5  # True: slope=3, intercept=5
    y = y_true + np.random.randn(100) * 2
    
    print("True model: y = 3x + 5")
    
    # Model WITH bias
    print("\n" + "-" * 60)
    print("Training model WITH bias term...")
    print("-" * 60)
    model_with_bias = ScratchLinearRegression(num_iter=500, lr=0.01, 
                                               no_bias=False, verbose=False)
    model_with_bias.fit(X, y)
    pred_with_bias = model_with_bias.predict(X)
    mse_with_bias = MSE(pred_with_bias, y)
    
    print(f"Learned: y = {model_with_bias.coef_[1]:.2f}x + {model_with_bias.coef_[0]:.2f}")
    print(f"MSE: {mse_with_bias:.4f}")
    
    # Model WITHOUT bias
    print("\n" + "-" * 60)
    print("Training model WITHOUT bias term...")
    print("-" * 60)
    model_no_bias = ScratchLinearRegression(num_iter=500, lr=0.01, 
                                             no_bias=True, verbose=False)
    model_no_bias.fit(X, y)
    pred_no_bias = model_no_bias.predict(X)
    mse_no_bias = MSE(pred_no_bias, y)
    
    print(f"Learned: y = {model_no_bias.coef_[0]:.2f}x (forced through origin)")
    print(f"MSE: {mse_no_bias:.4f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: Role of Bias Term")
    print("=" * 60)
    print(f"MSE with bias: {mse_with_bias:.4f}")
    print(f"MSE without bias: {mse_no_bias:.4f}")
    print(f"MSE increase without bias: {((mse_no_bias/mse_with_bias - 1) * 100):.1f}%")
    print("\nConclusion:")
    print("• Bias term θ₀ allows the model to fit data that doesn't pass through origin")
    print("• Without bias, the model is forced through (0,0), limiting flexibility")
    print("• For data with non-zero intercept, removing bias increases error significantly")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predictions comparison
    axes[0, 0].scatter(X, y, alpha=0.4, label='Data', s=20)
    axes[0, 0].plot(X, pred_with_bias, 'b-', linewidth=2, label='With Bias')
    axes[0, 0].plot(X, pred_no_bias, 'r--', linewidth=2, label='Without Bias')
    axes[0, 0].plot(X, y_true, 'g:', linewidth=2, label='True (y=3x+5)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Model Fit Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    axes[0, 1].plot(model_with_bias.loss, 'b-', linewidth=2, label='With Bias')
    axes[0, 1].plot(model_no_bias.loss, 'r-', linewidth=2, label='Without Bias')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss J(θ)')
    axes[0, 1].set_title('Learning Curves')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals with bias
    residuals_with = y - pred_with_bias
    axes[1, 0].scatter(pred_with_bias, residuals_with, alpha=0.5, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title(f'Residuals WITH Bias (MSE={mse_with_bias:.2f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals without bias
    residuals_without = y - pred_no_bias
    axes[1, 1].scatter(pred_no_bias, residuals_without, alpha=0.5, s=20, color='red')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'Residuals WITHOUT Bias (MSE={mse_no_bias:.2f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/problem8_bias_removal.png', dpi=300)
    
    print("\n✓ Analysis plot saved to plots/problem8_bias_removal.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
