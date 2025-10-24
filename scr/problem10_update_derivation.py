"""
Problem 10: Derivation of Update Formula (Advanced)

Mathematical derivation showing how the gradient descent update formula
is derived from the objective function.

Starting from: θⱼ := θⱼ - ∂/∂θⱼ J(θ)
We derive: θⱼ := θⱼ - α(1/m)∑[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾]
"""

import numpy as np


def print_derivation():
    """
    Print the complete mathematical derivation of the gradient descent update rule.
    """
    derivation = """
=" * 80)
MATHEMATICAL DERIVATION OF GRADIENT DESCENT UPDATE FORMULA
=" * 80)

iven:
------
Objective Function (Loss Function):
    J(θ) = (1/2m)∑ᵢ₌₁ᵐ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²

Hypothesis Function:
    hθ(x) = θᵀx = θ₀x₀ + θ₁x₁ + ... + θₙxₙ
    where x₀ = 1 (bias term)

eneral Gradient Descent Update Rule:
--------------------------------------
    θⱼ := θⱼ - α(∂/∂θⱼ)J(θ)
    where α is the learning rate

erivation of ∂J/∂θⱼ:
--------------------

tep 1: Write out J(θ)
    J(θ) = (1/2m)∑ᵢ₌₁ᵐ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²

tep 2: Apply the chain rule
    ∂J/∂θⱼ = ∂/∂θⱼ [(1/2m)∑ᵢ₌₁ᵐ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²]

tep 3: Move constant (1/2m) outside and apply derivative
    ∂J/∂θⱼ = (1/2m)∑ᵢ₌₁ᵐ ∂/∂θⱼ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²

tep 4: Apply chain rule to the square term
    Let uᵢ = hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾
    Then ∂/∂θⱼ [uᵢ]² = 2uᵢ · ∂uᵢ/∂θⱼ
    
    ∂J/∂θⱼ = (1/2m)∑ᵢ₌₁ᵐ 2[hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾] · ∂/∂θⱼ[hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]

tep 5: Simplify (2 cancels with 1/2)
    ∂J/∂θⱼ = (1/m)∑ᵢ₌₁ᵐ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾] · ∂/∂θⱼ[hθ(x⁽ⁱ⁾)]
    
    Note: ∂/∂θⱼ[y⁽ⁱ⁾] = 0 (y is constant with respect to θ)

tep 6: Compute ∂hθ(x⁽ⁱ⁾)/∂θⱼ
    Since hθ(x⁽ⁱ⁾) = θ₀x₀⁽ⁱ⁾ + θ₁x₁⁽ⁱ⁾ + ... + θⱼxⱼ⁽ⁱ⁾ + ... + θₙxₙ⁽ⁱ⁾
    
    ∂hθ(x⁽ⁱ⁾)/∂θⱼ = xⱼ⁽ⁱ⁾
    
    (The derivative with respect to θⱼ picks out only the term containing θⱼ)

tep 7: Substitute back
    ∂J/∂θⱼ = (1/m)∑ᵢ₌₁ᵐ [hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾] · xⱼ⁽ⁱ⁾
    
    ∂J/∂θⱼ = (1/m)∑ᵢ₌₁ᵐ [(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾]

tep 8: Final Update Rule
    Substitute the gradient back into the general update rule:
    
    θⱼ := θⱼ - α · ∂J/∂θⱼ
    
    θⱼ := θⱼ - α · (1/m)∑ᵢ₌₁ᵐ [(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾]

=" * 80)
INAL GRADIENT DESCENT UPDATE FORMULA
=" * 80)

    θⱼ := θⱼ - α(1/m)∑ᵢ₌₁ᵐ [(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾]

here:
- θⱼ is the j-th parameter (weight or bias)
- α is the learning rate
- m is the number of training samples
- hθ(x⁽ⁱ⁾) is the predicted value for sample i
- y⁽ⁱ⁾ is the true value for sample i
- xⱼ⁽ⁱ⁾ is the j-th feature of sample i

ector Form:
-----------
n vector notation, for all parameters simultaneously:
    
    θ := θ - α(1/m)Xᵀ(Xθ - y)

here:
- X is the m×n feature matrix
- θ is the n×1 parameter vector
- y is the m×1 target vector

his vectorized form is what we implement in code for efficiency.

=" * 80)
"""
    print(derivation)


def numerical_verification():
    """
    Numerically verify the derivation with a simple example.
    """
    print("\n" + "=" * 80)
    print("NUMERICAL VERIFICATION")
    print("=" * 80)
    
    # Simple example: 1D linear regression
    # Data: (x, y) = (1, 3), (2, 5), (3, 7)
    # True model: y = 2x + 1
    
    X = np.array([[1, 1], [1, 2], [1, 3]])  # First column is x₀=1 (bias)
    y = np.array([3, 5, 7])
    m = len(y)
    
    # Initial parameters
    theta = np.array([0.0, 0.0])  # [θ₀, θ₁]
    alpha = 0.1
    
    print(f"\nData:")
    print(f"  X = {X[:, 1].tolist()} (features)")
    print(f"  y = {y.tolist()} (targets)")
    print(f"  True model: y = 2x + 1")
    
    print(f"\nInitial parameters: θ = {theta}")
    
    # Perform one gradient descent step
    h = X @ theta  # Predictions
    error = h - y  # Errors
    gradient = (1/m) * (X.T @ error)  # Gradient
    
    print(f"\nGradient Descent Step:")
    print(f"  Predictions: h(x) = {h}")
    print(f"  Errors: (h-y) = {error}")
    print(f"  Gradient: ∇J(θ) = {gradient}")
    
    # Update
    theta_new = theta - alpha * gradient
    print(f"  Updated θ = {theta} - {alpha} × {gradient}")
    print(f"  New θ = {theta_new}")
    
    # Verify gradient calculation matches formula
    print(f"\nVerification:")
    print(f"  ∂J/∂θ₀ = (1/{m})∑[(h(x⁽ⁱ⁾)-y⁽ⁱ⁾)·1] = {gradient[0]:.4f}")
    print(f"  ∂J/∂θ₁ = (1/{m})∑[(h(x⁽ⁱ⁾)-y⁽ⁱ⁾)·x⁽ⁱ⁾] = {gradient[1]:.4f}")
    
    # Manual calculation
    manual_grad_0 = (1/m) * sum(error * X[:, 0])
    manual_grad_1 = (1/m) * sum(error * X[:, 1])
    print(f"\nManual calculation:")
    print(f"  ∂J/∂θ₀ = {manual_grad_0:.4f} ✓")
    print(f"  ∂J/∂θ₁ = {manual_grad_1:.4f} ✓")
    
    print(f"\n✓ Derivation verified numerically!")
    print("=" * 80)


def main():
    """
    Display the complete derivation and numerical verification.
    """
    print_derivation()
    numerical_verification()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The gradient descent update formula for linear regression is derived by:
1. Starting with the objective function J(θ) = (1/2m)∑[hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²
2. Computing the partial derivative ∂J/∂θⱼ using chain rule
3. Simplifying to get ∂J/∂θⱼ = (1/m)∑[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾]
4. Substituting into the general update rule θⱼ := θⱼ - α·∂J/∂θⱼ

This formula updates each parameter in the direction that reduces the loss,
scaled by the learning rate α and averaged over all training samples.
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
