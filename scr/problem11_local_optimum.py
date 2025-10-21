import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Problem 11 - Local Optimum Solutions
    Demonstrates convex cost surface (only one global minimum).
    """
    print("📉 Visualizing convex cost function surface...")

    theta0 = np.linspace(-5, 5, 100)
    theta1 = np.linspace(-5, 5, 100)
    T0, T1 = np.meshgrid(theta0, theta1)
    J = T0**2 + T1**2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T0, T1, J, cmap='viridis', alpha=0.9)
    ax.set_xlabel("θ₀")
    ax.set_ylabel("θ₁")
    ax.set_zlabel("J(θ)")
    ax.set_title("Convex Cost Function Surface")
    plt.savefig("plots/problem11_convex_surface.png")
    print("✅ Saved plot: plots/problem11_convex_surface.png")

if __name__ == "__main__":
    main()
