import numpy as np

class ScratchLinearRegression:
    """
    Linear regression prediction using the hypothesis function.
    """

    def __init__(self, no_bias=False):
        self.no_bias = no_bias
        self.coef_ = None

    def _linear_hypothesis(self, X):
        if self.coef_ is None:
            raise ValueError("Model coefficients not set.")

        if not self.no_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, self.coef_)

    def predict(self, X):
        """
        Return predicted values.
        """
        return self._linear_hypothesis(X)


def main():
    # Demo prediction using known weights
    model = ScratchLinearRegression(no_bias=False)
    model.coef_ = np.array([[2], [3]])  # bias=2, weight=3

    X_new = np.array([[1], [2], [3]])
    y_pred = model.predict(X_new)

    print("Predicted values:\n", y_pred)


if __name__ == "__main__":
    main()
