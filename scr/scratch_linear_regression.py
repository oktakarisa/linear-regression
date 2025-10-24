"""
Unified Scratch Linear Regression Implementation

This module contains the complete ScratchLinearRegression class that implements
linear regression from scratch using only NumPy.
"""

import numpy as np


class ScratchLinearRegression:
    """
    Scratch implementation of linear regression

    Parameters
    ----------
    num_iter : int
        Number of iterations for gradient descent
    lr : float
        Learning rate (alpha)
    no_bias : bool
        True if no bias term is included
    verbose : bool
        True to output the learning process

    Attributes
    ----------
    self.coef_ : ndarray, shape (n_features,)
        Parameters (weights and bias if included)
    self.loss : ndarray, shape (self.iter,)
        Record losses on training data
    self.val_loss : ndarray, shape (self.iter,)
        Record loss on validation data
    """

    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        """
        Initialize the linear regression model.
        """
        # Record hyperparameters as attributes
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        # Prepare an array to record the loss
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None

    def _linear_hypothesis(self, X):
        """
        Compute the linear hypothesis function for linear regression.
        
        The hypothesis function is defined as:
        hθ(x) = θ₀x₀ + θ₁x₁ + ... + θₙxₙ
        
        In vector form: hθ(x) = θᵀx
        
        where:
        - θᵢ are the model parameters (coefficients)
        - xᵢ are the feature values
        - x₀ = 1 (bias term, if included)
        - n is the number of features
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input features (already includes bias column if not self.no_bias)
            
        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted values using the linear hypothesis
        
        Notes
        -----
        When no_bias=False, x₀=1 should be prepended before calling this method.
        When no_bias=True, the model uses only the feature coefficients θ₁ through θₙ.
        """
        if self.coef_ is None:
            raise ValueError("Model coefficients not initialized. Call fit() first.")
        
        # Calculate hypothesis: hθ(x) = θᵀx
        return np.dot(X, self.coef_)

    def _gradient_descent(self, X, error):
        """
        Perform one step of gradient descent update.
        
        The update rule is:
        θⱼ := θⱼ - α(1/m)∑[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾]
        
        where:
        - α is the learning rate
        - m is the number of samples
        - hθ(x⁽ⁱ⁾) is the hypothesis for sample i
        - y⁽ⁱ⁾ is the true value for sample i
        - xⱼ⁽ⁱ⁾ is the j-th feature of sample i
        - error = hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training features (with bias column if applicable)
        error : ndarray, shape (n_samples,)
            Prediction errors (predictions - true values)
        """
        m = X.shape[0]  # Number of samples
        # Calculate gradient: ∂J/∂θ = (1/m)Xᵀ(error)
        gradient = np.dot(X.T, error) / m
        # Update parameters: θ := θ - α * gradient
        self.coef_ -= self.lr * gradient

    def _compute_loss(self, X, y):
        """
        Compute the objective function (loss function) for linear regression.
        
        J(θ) = (1/2m)∑[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²]
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features (with bias column if applicable)
        y : ndarray, shape (n_samples,)
            True target values
            
        Returns
        -------
        float
            Loss value
        """
        m = X.shape[0]
        predictions = self._linear_hypothesis(X)
        error = predictions - y
        # Objective function: J(θ) = (1/2m)∑(error²)
        loss = np.sum(error ** 2) / (2 * m)
        return loss

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Learn linear regression. If validation data is entered, the loss and 
        accuracy for it are also calculated for each iteration.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features of training data
        y : ndarray, shape (n_samples,)
            Correct answer value of training data
        X_val : ndarray, shape (n_samples, n_features), optional
            Features of verification data
        y_val : ndarray, shape (n_samples,), optional
            Correct value of verification data
        """
        # Input validation and conversion
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Add bias term if needed (x₀ = 1)
        if not self.no_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            if X_val is not None:
                X_val = np.c_[np.ones((X_val.shape[0], 1)), X_val]
        
        # Initialize parameters
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        
        # Training loop
        for i in range(self.iter):
            # Forward pass: compute predictions
            predictions = self._linear_hypothesis(X)
            error = predictions - y
            
            # Calculate and record training loss
            self.loss[i] = self._compute_loss(X, y)
            
            # Calculate validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                y_val = np.asarray(y_val).ravel()
                self.val_loss[i] = self._compute_loss(X_val, y_val)
            
            # Update parameters using gradient descent
            self._gradient_descent(X, error)
            
            # Print progress if verbose is True
            if self.verbose:
                step = max(1, self.iter // 10)
                if (i + 1) % step == 0:
                    if X_val is not None and y_val is not None:
                        print(f"Iteration {i+1}/{self.iter}, "
                              f"Training Loss: {self.loss[i]:.6f}, "
                              f"Validation Loss: {self.val_loss[i]:.6f}")
                    else:
                        print(f"Iteration {i+1}/{self.iter}, "
                              f"Training Loss: {self.loss[i]:.6f}")
        
        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final coefficients: {self.coef_}")

    def predict(self, X):
        """
        Estimate using linear regression.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Sample features
            
        Returns
        -------
        ndarray, shape (n_samples,)
            Estimated result by linear regression
        """
        X = np.asarray(X)
        
        # Add bias term if needed
        if not self.no_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Return predictions as 1-D array
        return self._linear_hypothesis(X).ravel()


def MSE(y_pred, y):
    """
    Calculate Mean Squared Error (MSE) between predicted and true values.
    
    The MSE is defined as:
    MSE = (1/m)∑[(y_pred⁽ⁱ⁾ - y⁽ⁱ⁾)²]
    
    where:
    - m is the number of samples
    - y_pred⁽ⁱ⁾ is the predicted value for sample i
    - y⁽ⁱ⁾ is the true value for sample i
    
    Parameters
    ----------
    y_pred : array-like
        Predicted values from the model
    y : array-like
        True target values
        
    Returns
    -------
    float
        Mean squared error value
        
    Raises
    ------
    ValueError
        If inputs have different shapes or contain NaN/infinite values
    """
    # Convert inputs to numpy arrays and flatten
    y_pred = np.asarray(y_pred).ravel()
    y = np.asarray(y).ravel()
    
    # Input validation
    if y_pred.shape != y.shape:
        raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y shape {y.shape}")
    
    if not np.isfinite(y_pred).all() or not np.isfinite(y).all():
        raise ValueError("Input contains NaN or infinite values")
    
    # Calculate MSE
    mse = np.mean((y_pred - y) ** 2)
    
    return mse

