import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Устанавливаем бэкенд TkAgg
import matplotlib.pyplot as plt


# 1) Euler's Method
def euler_method(f, x0, y0, h, n):
    """
    Implements Euler's method for solving the differential equation y' = f(x, y).

    :param f: Function representing the differential equation.
    :param x0: Initial x-value.
    :param y0: Initial y-value.
    :param h: Step size.
    :param n: Number of steps.
    :return: Arrays of x and y values.
    """
    x = np.linspace(x0, x0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])  # Euler's formula
    return x, y


# 2) Modified Euler's Method (Improved Euler)
def modified_euler_method(f, x0, y0, h, n):
    """
    Implements the improved Euler method (Heun's method).
    """
    x = np.linspace(x0, x0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)  # Predictor-corrector approach
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    return x, y


# 3) Runge-Kutta Method (3rd Order)
def runge_kutta_3(f, x0, y0, h, n):
    """
    Implements the 3rd-order Runge-Kutta method.
    """
    x = np.linspace(x0, x0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + (h / 2) * k1)
        k3 = f(x[i] + h, y[i] - h * k1 + 2 * h * k2)
        y[i + 1] = y[i] + (h / 6) * (k1 + 4 * k2 + k3)  # Weighted sum of slopes
    return x, y


# 3) Runge-Kutta Method (4th Order)
def runge_kutta_4(f, x0, y0, h, n):
    """
    Implements the 4th-order Runge-Kutta method.
    """
    x = np.linspace(x0, x0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + (h / 2) * k1)
        k3 = f(x[i] + h / 2, y[i] + (h / 2) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # More accurate estimate
    return x, y


# 4) Newton's Forward Difference Formula (1st and 2nd order derivatives)
def newton_forward_diff(y, h):
    """
    Computes the first and second derivatives using Newton's forward difference formula.

    :param y: Array of function values.
    :param h: Step size.
    :return: Lists of first and second derivatives.
    """
    n = len(y)
    first_derivative = [(y[i + 1] - y[i]) / h for i in range(n - 1)]  # First-order difference
    second_derivative = [(y[i + 2] - 2 * y[i + 1] + y[i]) / h ** 2 for i in range(n - 2)]  # Second-order difference
    return first_derivative, second_derivative


# Example differential equation
def func(x, y):
    return x + y  # Example equation y' = x + y


# Initial conditions
x0, y0, h, n = 0, 1, 0.1, 10  # x0 = start, y0 = initial value, h = step size, n = steps

# Solving using different methods
x_euler, y_euler = euler_method(func, x0, y0, h, n)
x_mod_euler, y_mod_euler = modified_euler_method(func, x0, y0, h, n)
x_rk3, y_rk3 = runge_kutta_3(func, x0, y0, h, n)
x_rk4, y_rk4 = runge_kutta_4(func, x0, y0, h, n)

# Compute derivatives using Newton's forward difference method
first_deriv, second_deriv = newton_forward_diff(y_rk4, h)

# Plotting the results
plt.plot(x_euler, y_euler, label="Euler Method")
plt.plot(x_mod_euler, y_mod_euler, label="Modified Euler")
plt.plot(x_rk3, y_rk3, label="Runge-Kutta 3rd Order")
plt.plot(x_rk4, y_rk4, label="Runge-Kutta 4th Order", linestyle="dashed")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Numerical Methods for Solving ODE")
plt.show()
