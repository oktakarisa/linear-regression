import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scr.problem5_objective_function import ScratchLinearRegression


def load_house_prices_data(filepath='data/train.csv'):
    """
    Load and preprocess the House Prices dataset from `data/train.csv`.
    If the file is not found, generate a synthetic dataset for demonstration.
    Returns X (DataFrame) and y (Series).
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset: {filepath} — shape={df.shape}")

        # Target
        if 'SalePrice' in df.columns:
            y = df['SalePrice']
            X = df.drop(columns=['SalePrice', 'Id'], errors='ignore')
        else:
            # fallback: numeric last column as target
            numeric = df.select_dtypes(include=[np.number])
            y = numeric.iloc[:, -1]
            X = numeric.iloc[:, :-1]

        # Keep only numerical features for the scratch implementation
        X = X.select_dtypes(include=[np.number])

        # Drop columns with >50% missing values
        missing_frac = X.isna().mean()
        drop_cols = missing_frac[missing_frac > 0.5].index.tolist()
        if drop_cols:
            print(f"Dropping columns with >50% missing: {drop_cols}")
            X = X.drop(columns=drop_cols)

        # Simple imputation: fill numeric NaNs with column mean
        X = X.fillna(X.mean())

        return X, y

    except FileNotFoundError:
        print("data/train.csv not found — using synthetic data for demonstration")
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'LotArea': np.random.normal(10000, 3000, n_samples),
            'YearBuilt': np.random.normal(1980, 20, n_samples),
            'BedroomAbvGr': np.random.normal(3, 1, n_samples),
            'GrLivArea': np.random.normal(1500, 500, n_samples)
        })
        y = (0.3 * X['LotArea'] + 0.4 * X['YearBuilt'] +
             0.2 * X['BedroomAbvGr'] + 0.5 * X['GrLivArea'] +
             np.random.normal(0, 1000, n_samples))
        return X, y


def compare_implementations(X, y, verbose=True):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scratch model
    # Add bias (intercept) column to inputs for scratch model if it expects it
    # Scratch implementation in scr/problem5_objective_function expects X to already include bias if used.
    X_train_scratch = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_scratch = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    # Scratch model
    model_scratch = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=verbose)
    model_scratch.fit(X_train_scratch, y_train)
    preds_scratch = model_scratch.predict(X_test_scratch)

    # sklearn model
    # Fit sklearn without intercept because scratch model includes bias as a column
    model_sklearn = LinearRegression(fit_intercept=False)
    model_sklearn.fit(X_train_scratch, y_train)
    preds_sklearn = model_sklearn.predict(X_test_scratch)

    results = {
        'scratch': {
            'mse': mean_squared_error(y_test, preds_scratch),
            'r2': r2_score(y_test, preds_scratch),
            'coef': model_scratch.coef_
        },
        'sklearn': {
            'mse': mean_squared_error(y_test, preds_sklearn),
            'r2': r2_score(y_test, preds_sklearn),
            'coef': model_sklearn.coef_
        }
    }

    # Return results and predictions for plotting
    return results, preds_scratch, preds_sklearn, y_test


def main():
    X, y = load_house_prices_data()
    results, preds_scratch, preds_sklearn, y_test = compare_implementations(X, y, verbose=False)

    print("\nComparison Results:")
    print("-" * 50)
    print("Scratch Implementation:")
    print(f"MSE: {results['scratch']['mse']:.2f}")
    print(f"R² Score: {results['scratch']['r2']:.4f}")
    print("\nScikit-learn Implementation:")
    print(f"MSE: {results['sklearn']['mse']:.2f}")
    print(f"R² Score: {results['sklearn']['r2']:.4f}")

    print("\nCoefficient Comparison:")
    print("-" * 50)
    scratch_coef = np.ravel(results['scratch']['coef'])
    sklearn_coef = np.ravel(results['sklearn']['coef'])

    # Print feature names with coefficients (feature 0 is bias/intercept)
    feature_names = ['bias'] + list(X.columns)
    for i in range(min(len(scratch_coef), len(sklearn_coef))):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        print(f"{name}: Scratch = {scratch_coef[i]:.4f}, sklearn = {sklearn_coef[i]:.4f}")

    # Create predicted vs actual plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Predictions on test set
    # Recompute predictions arrays in case of dtype shapes
    preds_scratch = np.asarray(preds_scratch).ravel()
    preds_sklearn = np.asarray(preds_sklearn).ravel()
    y_test_arr = np.asarray(y_test).ravel()

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test_arr, preds_scratch, alpha=0.5, label='Scratch')
    plt.scatter(y_test_arr, preds_sklearn, alpha=0.5, label='sklearn', marker='x')
    maxv = max(y_test_arr.max(), preds_sklearn.max(), preds_scratch.max())
    minv = min(y_test_arr.min(), preds_sklearn.min(), preds_scratch.min())
    plt.plot([minv, maxv], [minv, maxv], 'k--', linewidth=1)
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.title('Predicted vs Actual — Scratch vs sklearn')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    import os
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/problem6_comparison.png')
    print("Saved comparison plot to plots/problem6_comparison.png")


if __name__ == "__main__":
    main()
