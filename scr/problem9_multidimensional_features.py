import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scr.problem6_learning_estimation import ScratchLinearRegression

def main():
    # Generate nonlinear data
    X = np.linspace(-3, 3, 80).reshape(-1, 1)
    y = 0.5 * X.squeeze()**3 - X.squeeze()**2 + 2 * X.squeeze() + np.random.randn(80) * 3

    # Linear features
    model_linear = ScratchLinearRegression(num_iter=500, lr=0.001, no_bias=False, verbose=False)
    model_linear.fit(X, y)

    # Polynomial features (quadratic + cubic)
    X_poly = np.c_[X, X**2, X**3]
    model_poly = ScratchLinearRegression(num_iter=500, lr=0.001, no_bias=False, verbose=False)
    model_poly.fit(X_poly, y)

    # Plot losses
    plt.plot(model_linear.loss, label="Linear Features")
    plt.plot(model_poly.loss, label="Polynomial Features (x², x³)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Learning with Higher-Degree Features")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/problem9_multidimensional_features.png")
    print("Multidimensional feature comparison saved to plots/problem9_multidimensional_features.png")

if __name__ == "__main__":
    main()

