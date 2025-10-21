import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scr.problem6_learning_estimation import ScratchLinearRegression

def main():
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2.5 * X.squeeze() + np.random.randn(50) * 2

    model = ScratchLinearRegression(num_iter=200, lr=0.01, no_bias=False, verbose=False)
    model.fit(X, y)

    plt.plot(model.loss, label="Training Loss")
    if np.any(model.val_loss):
        plt.plot(model.val_loss, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/problem7_learning_curve.png")
    print("Learning curve plotted and saved to plots/problem7_learning_curve.png")

if __name__ == "__main__":
    main()