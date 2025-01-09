import math
def bisection_method(f, a, b, tol=1e-6, max_iter=7):
    if f(a) * f(b) >= 0:
        print("choose different values for a and b.")
        return None

    iteration_count = 0
    while (b - a) / 2 > tol and iteration_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iteration_count += 1
    return (a + b) / 2, iteration_count


def f(x):
    return math.exp(x)-x**2

root = bisection_method(f, -2, 0)
print("root:", root)
