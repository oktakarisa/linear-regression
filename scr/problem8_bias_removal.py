import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scr.problem6_learning_estimation import ScratchLinearRegression

def main():
    # Generate data not passing through origin
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3 * X.squeeze() + 5 + np.random.randn(50) * 2

    # Model with bias
    model_with_bias = ScratchLinearRegression(num_iter=200, lr=0.01, no_bias=False, verbose=False)
    model_with_bias.fit(X, y)

    # Model without bias
    model_no_bias = ScratchLinearRegression(num_iter=200, lr=0.01, no_bias=True, verbose=False)
    model_no_bias.fit(X, y)

    # Plot learning curves
    plt.plot(model_with_bias.loss, label="With Bias")
    plt.plot(model_no_bias.loss, label="No Bias")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Effect of Removing Bias Term")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/problem8_bias_removal.png")
    print("Bias removal comparison plotted and saved to plots/problem8_bias_removal.png")

if __name__ == "__main__":
    main()
