"""
Problem 6: Learning and Estimation

Compare the scratch implementation with scikit-learn's LinearRegression
on the House Prices dataset.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scr.scratch_linear_regression import ScratchLinearRegression
import os


def load_house_prices_data(filepath='data/train.csv'):
    """
    Load and preprocess the House Prices dataset.
    If not found, generate synthetic data.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {filepath} (shape={df.shape})")
        
        # Target variable
        if 'SalePrice' in df.columns:
            y = df['SalePrice']
            X = df.drop(columns=['SalePrice', 'Id'], errors='ignore')
        else:
            numeric = df.select_dtypes(include=[np.number])
            y = numeric.iloc[:, -1]
            X = numeric.iloc[:, :-1]
        
        # Keep only numerical features
        X = X.select_dtypes(include=[np.number])
        
        # Drop columns with >50% missing values
        missing_frac = X.isna().mean()
        drop_cols = missing_frac[missing_frac > 0.5].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} columns with >50% missing values")
            X = X.drop(columns=drop_cols)
        
        # Fill remaining NaN with column mean
        X = X.fillna(X.mean())
        
        print(f"  Final feature count: {X.shape[1]}")
        return X, y
        
    except FileNotFoundError:
        print("⚠ data/train.csv not found — generating synthetic data")
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
        print(f"  Generated {n_samples} synthetic samples with {X.shape[1]} features")
        return X, y


def main():
    """
    Compare scratch implementation with scikit-learn.
    """
    print("=" * 60)
    print("Problem 6: Learning and Estimation")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X, y = load_house_prices_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Train scratch model
    print("\n" + "-" * 60)
    print("Training Scratch Implementation...")
    print("-" * 60)
    model_scratch = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=False)
    model_scratch.fit(X_train_scaled, y_train)
    preds_scratch = model_scratch.predict(X_test_scaled)
    
    mse_scratch = mean_squared_error(y_test, preds_scratch)
    r2_scratch = r2_score(y_test, preds_scratch)
    
    print(f"Scratch MSE: {mse_scratch:.2f}")
    print(f"Scratch R² Score: {r2_scratch:.4f}")
    
    # Train scikit-learn model
    print("\n" + "-" * 60)
    print("Training Scikit-learn Implementation...")
    print("-" * 60)
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train_scaled, y_train)
    preds_sklearn = model_sklearn.predict(X_test_scaled)
    
    mse_sklearn = mean_squared_error(y_test, preds_sklearn)
    r2_sklearn = r2_score(y_test, preds_sklearn)
    
    print(f"Sklearn MSE: {mse_sklearn:.2f}")
    print(f"Sklearn R² Score: {r2_sklearn:.4f}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"MSE Difference: {abs(mse_scratch - mse_sklearn):.2f}")
    print(f"R² Difference: {abs(r2_scratch - r2_sklearn):.4f}")
    
    if abs(mse_scratch - mse_sklearn) < mse_sklearn * 0.01:
        print("✓ Implementations match closely (<1% difference)")
    else:
        print("⚠ Some difference between implementations")
    
    # Create comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, preds_scratch, alpha=0.5, label='Scratch')
    plt.scatter(y_test, preds_sklearn, alpha=0.5, marker='x', label='Sklearn')
    max_val = max(y_test.max(), preds_scratch.max(), preds_sklearn.max())
    min_val = min(y_test.min(), preds_scratch.min(), preds_sklearn.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    comparison_data = {
        'Scratch': [mse_scratch, r2_scratch],
        'Sklearn': [mse_sklearn, r2_sklearn]
    }
    x_pos = np.arange(2)
    width = 0.35
    plt.bar(x_pos - width/2, [mse_scratch/1000, r2_scratch], width, label='Scratch')
    plt.bar(x_pos + width/2, [mse_sklearn/1000, r2_sklearn], width, label='Sklearn')
    plt.xticks(x_pos, ['MSE (÷1000)', 'R²'])
    plt.ylabel('Value')
    plt.title('Metrics Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/problem6_comparison.png', dpi=300)
    print("\n✓ Comparison plot saved to plots/problem6_comparison.png")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
