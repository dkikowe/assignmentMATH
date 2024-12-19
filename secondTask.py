import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def f(x):
    return x**3 - x - 1

def df(x):
    return 3*x**2 - 1

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    x_list = [x0, x1]
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x_list.append(x2)
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return x_list

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x_list = [x0]
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        x_list.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x_list

def iteration_method(g, x0, tol=1e-6, max_iter=100):
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


secant_results = secant_method(f, x0, x1)
newton_results = newton_raphson(f, df, x0)
iteration_results = iteration_method(g, x0)


def calculate_errors(results):
    errors = [abs(results[i] - results[i - 1]) for i in range(1, len(results))]
    return errors

secant_errors = calculate_errors(secant_results)
newton_errors = calculate_errors(newton_results)
iteration_errors = calculate_errors(iteration_results)

print("Secant Method Results:")
print(secant_results)
print("Errors:", secant_errors)

print("\nNewton-Raphson Method Results:")
print(newton_results)
print("Errors:", newton_errors)

print("\nIteration Method Results:")
print(iteration_results)
print("Errors:", iteration_errors)

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