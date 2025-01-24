import numpy as np

# Trapezoidal Rule
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        integral += 2 * f(a + i * h)
    integral *= h / 2
    return integral

# Simpson's Rule
def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires n to be even.")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n, 2):
        integral += 2 * f(a + i * h)
    integral *= h / 3
    return integral

# Boole's Rule
def booles_rule(f, a, b, n):
    if n % 4 != 0:
        raise ValueError("Boole's rule requires n to be a multiple of 4.")
    h = (b - a) / n
    integral = 7 * (f(a) + f(b))
    for i in range(1, n, 4):
        integral += 32 * f(a + i * h) + 12 * f(a + (i + 1) * h) + 32 * f(a + (i + 2) * h) + 14 * f(a + (i + 3) * h)
    integral *= (2 * h) / 45
    return integral

# Weddle's Rule
def weddles_rule(f, a, b, n):
    if n % 6 != 0:
        raise ValueError("Weddle's rule requires n to be a multiple of 6.")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n, 6):
        integral += (
            5 * f(a + i * h) +
            1 * f(a + (i + 1) * h) +
            6 * f(a + (i + 2) * h) +
            1 * f(a + (i + 3) * h) +
            5 * f(a + (i + 4) * h) +
            f(a + (i + 5) * h)
        )
    integral *= (3 * h) / 10
    return integral

# Example Function
def f(x):
    return x**3 + 2 * x**2 + x + 1

# Integration bounds and number of subintervals
a, b = 0, 10
n = 6  # Choose n suitable for the specific rule (e.g., even for Simpson's, multiple of 6 for Weddle's)

# Perform the numerical integration
print(f"Trapezoidal Rule: {trapezoidal_rule(f, a, b, n):.6f}")
print(f"Simpson's Rule: {simpsons_rule(f, a, b, 6):.6f}")
print(f"Boole's Rule: {booles_rule(f, a, b, 12):.6f}")
print(f"Weddle's Rule: {weddles_rule(f, a, b, 6):.6f}")
