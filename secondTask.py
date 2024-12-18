import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    iter_count = 0
    while iter_count < max_iter:
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1 - fx0) < tol:
            print("Convergence criterion met.")
            return x1
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x2
        iter_count += 1
    return x1

def f1(x):
    return x**2 - x - 1

root_secant = secant_method(f1, 1, 2)
print("Secant method root:", root_secant)


def iteration_method(f, x0, tol=1e-6, max_iter=100):
    iter_count = 0
    while iter_count < max_iter:
        x1 = f(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
        iter_count += 1
    return x0

def f_iter(x):
    return (x + 1) / x

root_iter = iteration_method(f_iter, 1)
print("Iteration method root:", root_iter)

def newton_raphson_method(f, f_prime, x0, tol=1e-6, max_iter=100):
    iter_count = 0
    while iter_count < max_iter:
        fx, fx_prime = f(x0), f_prime(x0)
        if abs(fx) < tol:
            return x0
        x0 = x0 - fx / fx_prime
        iter_count += 1
    return x0

def f_prime1(x):
    return 2*x - 1

root_newton = newton_raphson_method(f1, f_prime1, 1)
print("Newton-Raphson method root:", root_newton)

methods = ['Secant', 'Iteration', 'Newton-Raphson']
roots = [root_secant, root_iter, root_newton]

plt.bar(range(len(methods)), roots, tick_label=methods, color=['blue', 'green', 'red'])
plt.xlabel('Methods')
plt.ylabel('Roots')
plt.title('Comparison of Numerical Methods')
plt.grid(True)
plt.show()

def absolute_error(x1, x2):
    return abs(x1 - x2)

def relative_error(x1, x2):
    return abs(x1 - x2) / abs(x1)


x_exact = 1.618
x_approx = root_secant
Ea = absolute_error(x_approx, x_exact)
Er = relative_error(x_approx, x_exact)
print("Absolute Error:", Ea)
print("Relative Error:", Er)
