"""
Problem 9: Multidimensional Features (Advanced)

Verify how learning results change when using polynomial features
(square and cube of the original feature).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scr.scratch_linear_regression import ScratchLinearRegression, MSE
import os


def main():
    """
    Compare linear features vs polynomial features.
    """
    print("=" * 60)
    print("Problem 9: Multidimensional Features (Advanced)")
    print("=" * 60)
    
    # Generate nonlinear data (cubic function)
    print("\nGenerating nonlinear data (y = 0.5x³ - x² + 2x + noise)...")
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = 0.5 * X.squeeze()**3 - X.squeeze()**2 + 2 * X.squeeze()
    y = y_true + np.random.randn(100) * 3
    
    print("True model: y = 0.5x³ - x² + 2x")
    
    # Model 1: Linear features only
    print("\n" + "-" * 60)
    print("Model 1: Linear features only")
    print("-" * 60)
    model_linear = ScratchLinearRegression(num_iter=1000, lr=0.001, 
                                           no_bias=False, verbose=False)
    model_linear.fit(X, y)
    pred_linear = model_linear.predict(X)
    mse_linear = MSE(pred_linear, y)
    
    print(f"Features: [x]")
    print(f"MSE: {mse_linear:.4f}")
    print(f"Final loss: {model_linear.loss[-1]:.6f}")
    
    # Model 2: Polynomial features (x, x², x³)
    print("\n" + "-" * 60)
    print("Model 2: Polynomial features (x, x², x³)")
    print("-" * 60)
    X_poly = np.c_[X, X**2, X**3]
    model_poly = ScratchLinearRegression(num_iter=1000, lr=0.001, 
                                         no_bias=False, verbose=False)
    model_poly.fit(X_poly, y)
    pred_poly = model_poly.predict(X_poly)
    mse_poly = MSE(pred_poly, y)
    
    print(f"Features: [x, x², x³]")
    print(f"MSE: {mse_poly:.4f}")
    print(f"Final loss: {model_poly.loss[-1]:.6f}")
    print(f"Coefficients: θ₀={model_poly.coef_[0]:.3f}, "
          f"θ₁={model_poly.coef_[1]:.3f}, "
          f"θ₂={model_poly.coef_[2]:.3f}, "
          f"θ₃={model_poly.coef_[3]:.3f}")
    print(f"Compare to true: [intercept≈0, θ₁≈2, θ₂≈-1, θ₃≈0.5]")
    
    # Model 3: Quadratic features only (x, x²)
    print("\n" + "-" * 60)
    print("Model 3: Quadratic features (x, x²)")
    print("-" * 60)
    X_quad = np.c_[X, X**2]
    model_quad = ScratchLinearRegression(num_iter=1000, lr=0.001, 
                                         no_bias=False, verbose=False)
    model_quad.fit(X_quad, y)
    pred_quad = model_quad.predict(X_quad)
    mse_quad = MSE(pred_quad, y)
    
    print(f"Features: [x, x²]")
    print(f"MSE: {mse_quad:.4f}")
    print(f"Final loss: {model_quad.loss[-1]:.6f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: Impact of Polynomial Features")
    print("=" * 60)
    print(f"Linear model MSE: {mse_linear:.4f}")
    print(f"Quadratic model MSE: {mse_quad:.4f} ({((mse_linear-mse_quad)/mse_linear*100):.1f}% improvement)")
    print(f"Cubic model MSE: {mse_poly:.4f} ({((mse_linear-mse_poly)/mse_linear*100):.1f}% improvement)")
    print("\nConclusion:")
    print("• Polynomial features allow linear regression to fit nonlinear patterns")
    print("• Matching the degree to the true function gives best results")
    print("• Higher-degree features increase model expressiveness")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort X for smooth plotting
    sort_idx = X.squeeze().argsort()
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    y_true_sorted = y_true[sort_idx]
    pred_linear_sorted = pred_linear[sort_idx]
    pred_quad_sorted = pred_quad[sort_idx]
    pred_poly_sorted = pred_poly[sort_idx]
    
    # Plot 1: All models comparison
    axes[0, 0].scatter(X_sorted, y_sorted, alpha=0.3, s=20, label='Data', color='gray')
    axes[0, 0].plot(X_sorted, y_true_sorted, 'k-', linewidth=2, label='True Function', alpha=0.7)
    axes[0, 0].plot(X_sorted, pred_linear_sorted, 'b--', linewidth=2, label='Linear')
    axes[0, 0].plot(X_sorted, pred_quad_sorted, 'g-.', linewidth=2, label='Quadratic')
    axes[0, 0].plot(X_sorted, pred_poly_sorted, 'r-', linewidth=2, label='Cubic')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Model Fit Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    axes[0, 1].plot(model_linear.loss, 'b-', linewidth=2, label='Linear', alpha=0.8)
    axes[0, 1].plot(model_quad.loss, 'g-', linewidth=2, label='Quadratic', alpha=0.8)
    axes[0, 1].plot(model_poly.loss, 'r-', linewidth=2, label='Cubic', alpha=0.8)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss J(θ)')
    axes[0, 1].set_title('Learning Curves')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MSE comparison
    models = ['Linear\n(x)', 'Quadratic\n(x, x²)', 'Cubic\n(x, x², x³)']
    mses = [mse_linear, mse_quad, mse_poly]
    colors = ['blue', 'green', 'red']
    axes[1, 0].bar(models, mses, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Mean Squared Error Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, (model, mse_val) in enumerate(zip(models, mses)):
        axes[1, 0].text(i, mse_val + 5, f'{mse_val:.1f}', ha='center', va='bottom')
    
    # Plot 4: Residuals for cubic model
    residuals = y - pred_poly
    axes[1, 1].scatter(pred_poly, residuals, alpha=0.5, s=20, color='red')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'Residuals: Cubic Model (MSE={mse_poly:.2f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/problem9_multidimensional_features.png', dpi=300)
    
    print("\n✓ Analysis plot saved to plots/problem9_multidimensional_features.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
