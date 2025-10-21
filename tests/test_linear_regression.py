import unittest
import numpy as np
from scr.problem5_objective_function import ScratchLinearRegression

class TestScratchLinearRegression(unittest.TestCase):
    def test_learns_simple_linear(self):
        """Test if model can learn a simple linear relationship."""
        # Generate simple linear data: y = 2x + 1
        X = np.array([[x] for x in range(10)])
        y = 2 * X.flatten() + 1 + np.random.normal(0, 0.1, 10)
        
        # Add bias column
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        model = ScratchLinearRegression(num_iter=1000, lr=0.01)
        model.fit(X_with_bias, y)
        y_pred = model.predict(X_with_bias)
        
        # Check if coefficients are close to true values
        self.assertAlmostEqual(model.coef_[1], 2, delta=0.2)  # slope
        self.assertAlmostEqual(model.coef_[0], 1, delta=0.2)  # intercept
        
        # Check if predictions are reasonable
        mse = np.mean((y - y_pred) ** 2)
        self.assertLess(mse, 0.5)

    def test_handles_no_bias(self):
        """Test if model works correctly with no_bias=True."""
        X = np.array([[x] for x in range(10)])
        y = 2 * X.flatten() + np.random.normal(0, 0.1, 10)
        
        model = ScratchLinearRegression(num_iter=1000, lr=0.01, no_bias=True)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Check that coef_ has correct shape (no bias term)
        self.assertEqual(len(model.coef_), X.shape[1])
        
        # Check if predictions are reasonable
        mse = np.mean((y - y_pred) ** 2)
        self.assertLess(mse, 0.5)

    def test_handles_multiple_features(self):
        """Test if model can handle multiple features."""
        # Generate data with two features: y = x1 + 2x2 + 1
        X = np.random.rand(100, 2)
        y = X[:, 0] + 2 * X[:, 1] + 1 + np.random.normal(0, 0.1, 100)
        
        # Add bias column
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        model = ScratchLinearRegression(num_iter=2000, lr=0.005)
        model.fit(X_with_bias, y)
        y_pred = model.predict(X_with_bias)
        
        # Check coefficients (with wider tolerance due to random data)
        self.assertAlmostEqual(model.coef_[0], 1, delta=0.5)  # bias
        self.assertAlmostEqual(model.coef_[1], 1, delta=0.5)  # x1 coefficient
        self.assertAlmostEqual(model.coef_[2], 2, delta=0.5)  # x2 coefficient
        
        # Check predictions
        mse = np.mean((y - y_pred) ** 2)
        self.assertLess(mse, 0.5)

if __name__ == '__main__':
    unittest.main()