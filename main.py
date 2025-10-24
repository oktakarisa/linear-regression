#!/usr/bin/env python3
"""
Main entry point for the Scratch Linear Regression assignment.

Runs all problem scripts in order (1 - 11), printing progress and handling errors gracefully.
"""

import importlib
import os
import sys
from tqdm import tqdm

# Folder paths
SCR_DIR = os.path.join(os.getcwd(), "scr")
DATA_DIR = os.path.join(os.getcwd(), "data")
PLOTS_DIR = os.path.join(os.getcwd(), "plots")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Problem scripts to run (in order)
PROBLEMS = [
    "problem1_hypothesis_function",
    "problem2_gradient_descent",
    "problem3_prediction",
    "problem4_mean_squared_error",
    "problem5_objective_function",
    "problem6_learning_estimation",
    "problem7_learning_curve_plot",
    "problem8_bias_removal",
    "problem9_multidimensional_features",
    "problem10_update_derivation",
    "problem11_local_optimum"
]

def main():
    print("\n" + "="*60)
    print("Starting Scratch Linear Regression Assignment Runner")
    print("="*60 + "\n")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")
    print("-"*60 + "\n")

    for script_name in tqdm(PROBLEMS, desc="Running Problems"):
        try:
            module_path = f"scr.{script_name}"
            module = importlib.import_module(module_path)

            if hasattr(module, "main"):
                print(f"\n> Running {script_name}.py ...")
                module.main()
                print(f"[OK] Completed: {script_name}\n")
            else:
                print(f"[SKIP] {script_name}: no main() function found.\n")

        except ModuleNotFoundError:
            print(f"[ERROR] {script_name}.py not found - skipping.\n")
        except Exception as e:
            print(f"[ERROR] Error running {script_name}: {e}\n")

    print("-"*60)
    print("All available problem scripts processed successfully.")
    print("Check the 'plots' folder for generated figures.\n")

if __name__ == "__main__":
    main()
