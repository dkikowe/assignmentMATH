import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def f(x):
    return x**3 - x - 1

def df(x):
    return 3*x**2 - 1

def secant_method(f, x0, x1, tol=0.000001, max_iter=100):
    x_list = [x0, x1]
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x_list.append(x2)
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return x_list

def newton_raphson(f, df, x0, tol=0.000001, max_iter=100):
    x_list = [x0]
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        x_list.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x_list

def iteration_method(g, x0, tol=0.000001, max_iter=100):
    x_list = [x0]
    for _ in range(max_iter):
        x1 = g(x0)
        x_list.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x_list

def g(x):
    return (x + 1)**(1/3)

x0 = 1.5
x1 = 1.6
true_root = 1.325

secant_results = secant_method(f, x0, x1)
newton_results = newton_raphson(f, df, x0)
iteration_results = iteration_method(g, x0)


def calculate_absolute_errors(results, true_value):
    return [abs(res - true_value) for res in results]

def calculate_relative_errors(abs_errors, true_value):
    return [abs_err / abs(true_value) for abs_err in abs_errors]

secant_abs_errors = calculate_absolute_errors(secant_results, true_root)
newton_abs_errors = calculate_absolute_errors(newton_results, true_root)
iteration_abs_errors = calculate_absolute_errors(iteration_results, true_root)

secant_rel_errors = calculate_relative_errors(secant_abs_errors, true_root)
newton_rel_errors = calculate_relative_errors(newton_abs_errors, true_root)
iteration_rel_errors = calculate_relative_errors(iteration_abs_errors, true_root)

for method, results, abs_errors, rel_errors in [
    ("Secant Method", secant_results, secant_abs_errors, secant_rel_errors),
    ("Newton-Raphson Method", newton_results, newton_abs_errors, newton_rel_errors),
    ("Iteration Method", iteration_results, iteration_abs_errors, iteration_rel_errors),
]:
    print(f"\n{method} Results:")
    for i, (res, abs_err, rel_err) in enumerate(zip(results, abs_errors, rel_errors)):
        print(f"Iteration {i}: Value = {res:.6f}, Absolute Error = {abs_err:.6f}, Relative Error = {rel_err:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(range(len(secant_results)), secant_results, label='Secant Method', marker='o')
plt.plot(range(len(newton_results)), newton_results, label='Newton-Raphson Method', marker='x')
plt.plot(range(len(iteration_results)), iteration_results, label='Iteration Method', marker='s')
plt.title('Root Finding Methods Comparison')
plt.xlabel('Iteration')
plt.ylabel('Value of Root')
plt.axhline(0, color='black', lw=1, ls='--')
plt.legend()
plt.grid()
plt.show()