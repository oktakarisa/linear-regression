import numpy as np

class ScratchLinearRegression:
    """
    Scratch implementation of linear regression with gradient descent.
    """

    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.coef_ = None

    def _linear_hypothesis(self, X):
        if self.coef_ is None:
            raise ValueError("Model coefficients not initialized.")

        if not self.no_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, self.coef_)

    def _gradient_descent(self, X, error):
        """
        Performs one step of gradient descent update:
        θ := θ - (α/m) * Xᵀ(error)
        """
        m = X.shape[0]
        grad = np.dot(X.T, error) / m
        self.coef_ -= self.lr * grad

    def fit(self, X, y):
        """
        Train model using gradient descent.
        """
        if not self.no_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        n_features = X.shape[1]
        self.coef_ = np.zeros((n_features, 1))

        for i in range(self.iter):
            predictions = np.dot(X, self.coef_)
            error = predictions - y
            self._gradient_descent(X, error)

        if self.verbose:
            print("Final coefficients:\n", self.coef_)


def main():
    # Simple test case
    X = np.array([[1], [2], [3]])
    y = np.array([[2], [4], [6]])

    model = ScratchLinearRegression(num_iter=1000, lr=0.1, verbose=True)
    model.fit(X, y)
    print("Learned coefficients:\n", model.coef_)


if __name__ == "__main__":
    main()
