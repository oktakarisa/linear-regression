import numpy as np

class ScratchLinearRegression:
    """
    Scratch implementation of linear regression
    """

    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None  # weights (and bias if included)

    def _linear_hypothesis(self, X):
        """
        Compute the linear hypothesis function:
        hθ(x) = θ₀ + θ₁x₁ + ... + θₙxₙ  (if bias included)
        or
        hθ(x) = θ₁x₁ + ... + θₙxₙ  (if bias excluded)
        """
        if self.coef_ is None:
            raise ValueError("Model coefficients not initialized. Fit or set self.coef_ first.")

        if not self.no_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, self.coef_)


def main():
    model = ScratchLinearRegression(no_bias=False)
    model.coef_ = np.array([[2], [3]])  # bias = 2, weight = 3
    X = np.array([[1], [2], [3]])
    print("Predictions:\n", model._linear_hypothesis(X))


if __name__ == "__main__":
    main()
