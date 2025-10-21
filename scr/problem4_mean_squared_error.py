import numpy as np

def MSE(y_pred, y):
    """
    Calculate Mean Squared Error (MSE)
    """
    mse = np.mean((y_pred - y) ** 2)
    return mse


def main():
    # Simple test
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    print("MSE:", MSE(y_pred, y_true))


if __name__ == "__main__":
    main()
