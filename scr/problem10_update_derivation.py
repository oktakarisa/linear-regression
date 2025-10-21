import sympy as sp

def main():
    """
    Problem 10 - Derivation of Gradient Descent Update Formula
    """
    print("Deriving Gradient Descent Update Formula...")

    m, alpha = sp.symbols('m alpha', real=True, positive=True)
    x, y = sp.symbols('x y')
    theta0, theta1 = sp.symbols('theta0 theta1')
    X = sp.Matrix([[1, x]])
    theta = sp.Matrix([[theta0], [theta1]])

    h = X * theta
    h_scalar = h[0]   # extract the scalar value
    J = (1 / (2*m)) * (h_scalar - y)**2

    grad_theta1 = sp.diff(J, theta1)
    print("∂J/∂θ₁ =", grad_theta1)
    print("Update rule: θ₁ := θ₁ - α * (1/m) * Σ((hθ(x) - y) * x₁)")
    print("Derivation complete.")

if __name__ == "__main__":
    main()
