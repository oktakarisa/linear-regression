import numpy as np
import unittest
from scr.problem5_objective_function import ScratchLinearRegression

class TestScratchLinearRegression(unittest.TestCase):
    def test_learns_simple_linear(self):
        # y = 2x + 1
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = 2 * X.squeeze() + 1

        # Add bias column
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        model = ScratchLinearRegression(num_iter=2000, lr=0.01, verbose=False)
        model.fit(X_bias, y)
        preds = model.predict(X_bias).ravel()

        mse = np.mean((preds - y) ** 2)
        # Expect low MSE
        self.assertLess(mse, 1e-1)

if __name__ == '__main__':
    unittest.main()
