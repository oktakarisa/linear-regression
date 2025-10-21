import numpy as np
from scr.problem4_mean_squared_error import MSE

class ScratchLinearRegression:
    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def _linear_hypothesis(self, X):
        return np.dot(X, self.coef_)

    def fit(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        self.coef_ = np.zeros(n)

        for i in range(self.iter):
            y_pred = self._linear_hypothesis(X)
            error = y_pred - y
            self.coef_ -= self.lr * (1/m) * np.dot(X.T, error)

            # Loss calculation
            self.loss[i] = (1/(2*m)) * np.sum(error ** 2)
            if X_val is not None and y_val is not None:
                val_pred = self._linear_hypothesis(X_val)
                val_error = val_pred - y_val
                self.val_loss[i] = (1/(2*m)) * np.sum(val_error ** 2)

            if self.verbose and i % 100 == 0:
                print(f"Iter {i}: Loss={self.loss[i]:.4f}")

    def predict(self, X):
        return self._linear_hypothesis(X)


def main():
    # Simple test for training and evaluation
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # Perfect linear relation: y = 2x
    model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    print("Final coefficients:", model.coef_)
    print("Training MSE:", MSE(y_pred, y))


if __name__ == "__main__":
    main()
