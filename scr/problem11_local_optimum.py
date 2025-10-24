"""
Problem 11: Local Optimum Problem (Advanced)

Explains why linear regression with gradient descent does NOT suffer from
local optima - it always converges to the global optimum (given appropriate
learning rate and sufficient iterations).

This is due to the CONVEXITY of the linear regression objective function.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def mathematical_explanation():
    """
    Print the mathematical explanation of why linear regression has no local optima.
    """
    explanation = """
=" * 80)
HY LINEAR REGRESSION HAS NO LOCAL OPTIMA
=" * 80)

bjective Function for Linear Regression:
-----------------------------------------
    J(θ) = (1/2m)∑ᵢ₌₁ᵐ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²
    
here hθ(x) = θᵀx (linear hypothesis)

ey Mathematical Property: CONVEXITY
-----------------------------------

definition, a function f(θ) is CONVEX if for any two points θ₁ and θ₂,
nd any λ ∈ [0,1]:
    
    f(λθ₁ + (1-λ)θ₂) ≤ λf(θ₁) + (1-λ)f(θ₂)

his means the function "curves upward" - any line segment between two points
n the function lies above the function itself.

roof that J(θ) is Convex:
-------------------------

tep 1: Expand J(θ) in vector form
    J(θ) = (1/2m)(Xθ - y)ᵀ(Xθ - y)
    J(θ) = (1/2m)[θᵀXᵀXθ - 2θᵀXᵀy + yᵀy]

tep 2: Compute the Hessian matrix (second derivatives)
    The Hessian H is the matrix of second partial derivatives:
    
    H = ∂²J/∂θ∂θᵀ = (1/m)XᵀX

tep 3: Verify positive semi-definiteness
    For any vector v ≠ 0:
    
    vᵀHv = (1/m)vᵀ(XᵀX)v = (1/m)(Xv)ᵀ(Xv) = (1/m)||Xv||² ≥ 0
    
    Since vᵀHv ≥ 0 for all v, the Hessian H is positive semi-definite.

onclusion:
---------
function whose Hessian is positive semi-definite everywhere is CONVEX.
herefore, J(θ) is a convex function.

mplications of Convexity:
-------------------------
. UNIQUE GLOBAL MINIMUM: A convex function has at most one minimum,
  which is the global minimum.

. NO LOCAL MINIMA: There are no "valleys" or local minima where gradient
  descent could get stuck.

. CONVERGENCE GUARANTEE: Starting from any initial point, gradient descent
  will converge to the global minimum (with appropriate learning rate).

. BOWL-SHAPED SURFACE: The cost function forms a "bowl" or "parabola"
  shape with a single minimum at the bottom.

raphical Intuition:
-------------------
or 2 parameters (θ₀, θ₁), the cost surface J(θ) looks like a bowl:
- Any path downhill leads to the same minimum
- No "bumps" or local valleys exist
- The gradient always points toward the global minimum

his is fundamentally different from non-convex functions (e.g., neural networks)
here multiple local minima can trap gradient descent.

athematical Proof of Global Convergence:
-----------------------------------------
ince ∇J(θ*) = 0 at the minimum θ*, and J is convex:
    
    J(θ) ≥ J(θ*) + ∇J(θ*)ᵀ(θ - θ*)
    J(θ) ≥ J(θ*)  (since ∇J(θ*) = 0)
    
his proves θ* is the GLOBAL minimum - no other point can have lower cost.

=" * 80)
UMMARY
=" * 80)

inear regression ALWAYS finds the optimal solution because:
✓ The objective function J(θ) is convex (bowl-shaped)
✓ Convexity guarantees a unique global minimum
✓ Gradient descent always converges to this global minimum
✓ No local minima exist to trap the optimization

his is a major advantage of linear regression over more complex models!

=" * 80)
"""
    print(explanation)


def visualize_convex_surface():
    """
    Create visualizations showing the convex cost surface.
    """
    print("\n" + "=" * 80)
    print("VISUALIZING CONVEX COST SURFACE")
    print("=" * 80)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ============================================
    # Plot 1: 3D Surface - Convex (Bowl-shaped)
    # ============================================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    theta0 = np.linspace(-5, 5, 100)
    theta1 = np.linspace(-5, 5, 100)
    T0, T1 = np.meshgrid(theta0, theta1)
    J_convex = T0**2 + T1**2  # Convex: always curves upward
    
    surf1 = ax1.plot_surface(T0, T1, J_convex, cmap='viridis', alpha=0.9, 
                             edgecolor='none', antialiased=True)
    ax1.set_xlabel('θ₀', fontsize=10)
    ax1.set_ylabel('θ₁', fontsize=10)
    ax1.set_zlabel('J(θ)', fontsize=10)
    ax1.set_title('Linear Regression Cost Surface\n(CONVEX - Bowl Shaped)', 
                  fontsize=11, pad=15)
    ax1.view_init(elev=30, azim=45)
    
    # Mark the global minimum
    ax1.scatter([0], [0], [0], color='red', s=100, marker='*', 
                label='Global Minimum')
    
    # ============================================
    # Plot 2: Contour of Convex Function
    # ============================================
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contour(T0, T1, J_convex, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(0, 0, 'r*', markersize=15, label='Global Minimum')
    
    # Show gradient descent paths from different starting points
    np.random.seed(42)
    for _ in range(5):
        start = np.random.uniform(-4, 4, 2)
        path = [start]
        theta_current = start.copy()
        lr = 0.1
        for _ in range(30):
            gradient = 2 * theta_current  # Gradient of θ₀² + θ₁²
            theta_current = theta_current - lr * gradient
            path.append(theta_current.copy())
        path = np.array(path)
        ax2.plot(path[:, 0], path[:, 1], 'w-', alpha=0.6, linewidth=1)
        ax2.plot(path[0, 0], path[0, 1], 'wo', markersize=4)
    
    ax2.set_xlabel('θ₀')
    ax2.set_ylabel('θ₁')
    ax2.set_title('Contour Plot: All Paths Lead to Global Minimum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ============================================
    # Plot 3: Cross-section showing convexity
    # ============================================
    ax3 = fig.add_subplot(2, 3, 3)
    theta_range = np.linspace(-5, 5, 100)
    J_1d = theta_range**2
    ax3.plot(theta_range, J_1d, 'b-', linewidth=2, label='J(θ)')
    ax3.plot(0, 0, 'r*', markersize=15, label='Global Minimum')
    
    # Show that any line segment lies above the curve (convexity property)
    theta_a, theta_b = -3, 4
    J_a, J_b = theta_a**2, theta_b**2
    ax3.plot([theta_a, theta_b], [J_a, J_b], 'g--', linewidth=2, 
             label='Line segment')
    ax3.plot([theta_a, theta_b], [J_a, J_b], 'go', markersize=8)
    
    ax3.set_xlabel('θ')
    ax3.set_ylabel('J(θ)')
    ax3.set_title('Cross-Section: Line Segment Above Curve\n(Convexity Property)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ============================================
    # Plot 4: 3D Non-convex (for comparison)
    # ============================================
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    J_nonconvex = np.sin(T0) * np.cos(T1) + 0.1 * (T0**2 + T1**2)
    
    surf2 = ax4.plot_surface(T0, T1, J_nonconvex, cmap='plasma', alpha=0.9,
                             edgecolor='none', antialiased=True)
    ax4.set_xlabel('θ₀', fontsize=10)
    ax4.set_ylabel('θ₁', fontsize=10)
    ax4.set_zlabel('J(θ)', fontsize=10)
    ax4.set_title('Non-Convex Function (for comparison)\n(Has Local Minima)', 
                  fontsize=11, pad=15)
    ax4.view_init(elev=30, azim=45)
    
    # ============================================
    # Plot 5: Hessian eigenvalues (proves convexity)
    # ============================================
    ax5 = fig.add_subplot(2, 3, 5)
    
    # For J(θ) = θ₀² + θ₁², Hessian = [[2, 0], [0, 2]]
    # Eigenvalues are both 2 > 0, proving positive definiteness
    
    H_example = np.array([[2, 0], [0, 2]])
    eigenvalues = np.linalg.eigvals(H_example)
    
    ax5.bar(['λ₁', 'λ₂'], eigenvalues, color=['blue', 'green'], alpha=0.7)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_ylabel('Eigenvalue')
    ax5.set_title('Hessian Eigenvalues\n(All positive → Convex)')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.text(0.5, max(eigenvalues)*0.5, 
             'All eigenvalues > 0\n⇒ Positive Definite\n⇒ CONVEX', 
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ============================================
    # Plot 6: Gradient descent convergence
    # ============================================
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Simulate gradient descent
    theta_init = np.array([4.5, -3.5])
    theta_history = [theta_init]
    loss_history = [np.sum(theta_init**2)]
    theta_current = theta_init.copy()
    lr = 0.1
    
    for i in range(50):
        gradient = 2 * theta_current
        theta_current = theta_current - lr * gradient
        theta_history.append(theta_current.copy())
        loss_history.append(np.sum(theta_current**2))
    
    ax6.plot(loss_history, 'b-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Loss J(θ)')
    ax6.set_title('Gradient Descent Convergence\n(Monotonic decrease to global minimum)')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/problem11_convex_surface.png', dpi=300, bbox_inches='tight')
    
    print("\n✓ Visualization saved to plots/problem11_convex_surface.png")
    print("  The plots demonstrate:")
    print("  • Bowl-shaped cost surface (no bumps or valleys)")
    print("  • All gradient descent paths converge to the same point")
    print("  • Positive Hessian eigenvalues (mathematical proof)")
    print("  • Monotonic loss decrease during optimization")
    
    return fig


def main():
    """
    Complete explanation of why linear regression has no local optima.
    """
    print("\n")
    mathematical_explanation()
    visualize_convex_surface()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Linear regression's objective function J(θ) = (1/2m)∑[hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]² is CONVEX.

Mathematical Proof:
1. The Hessian H = (1/m)XᵀX is positive semi-definite
2. For any vector v: vᵀHv = (1/m)||Xv||² ≥ 0
3. A function with positive semi-definite Hessian is convex

Practical Implications:
✓ Gradient descent ALWAYS finds the global optimum
✓ No risk of getting stuck in local minima
✓ Initial parameter values don't affect final solution (only convergence speed)
✓ This is a huge advantage over non-convex optimization (e.g., neural networks)

This convexity property makes linear regression reliable and predictable,
guaranteeing optimal solutions regardless of initialization.
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
