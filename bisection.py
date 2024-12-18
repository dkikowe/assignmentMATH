def bisection_method(f, a, b, tol=1e-6, max_iter=7):
    if f(a) * f(b) >= 0:
        print("choose different values for a and b.")
        return None

    iter_count = 0
    while (b - a) / 2 > tol and iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return (a + b) / 2


def f(x):
    return x**3 - 4*x - 9

root = bisection_method(f, -2, 0)
print("Root:", root)
