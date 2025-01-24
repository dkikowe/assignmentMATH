def newtons_forward_interpolation(x, fx, target_x):
    n = len(x)

    # Checking for uniformity of steps
    h = x[1] - x[0]  # Assuming uniform spacing
    for i in range(1, n - 1):
        if (x[i + 1] - x[i]) != h:
            return "Newton Forward Interpolation is not suitable for non-uniform nodes."

    forward_diff = [[0] * n for _ in range(n)]  # Table of forward differences

    # Fill the first column with fx
    for i in range(n):
        forward_diff[i][0] = fx[i]

    # Calculate forward differences
    for j in range(1, n):
        for i in range(n - j):
            forward_diff[i][j] = forward_diff[i + 1][j - 1] - forward_diff[i][j - 1]

    # Calculate p
    p = (target_x - x[0]) / h

    # Apply Newton's forward interpolation formula
    result = fx[0]
    term = 1
    for i in range(1, n):
        term *= (p - (i - 1)) / i
        result += term * forward_diff[0][i]

    return result

def lagranges_interpolation(x, fx, target_x):
    n = len(x)

    h = x[1] - x[0]
    for i in range(1, n - 1):
        if (x[i + 1] - x[i]) == h:
            return "Lagrange's Interpolation is not suitable for uniform nodes."

    result = 0

    # Compute Lagrange basis polynomials and sum them
    for i in range(n):
        term = fx[i]
        for j in range(n):
            if i != j:
                term *= (target_x - x[j]) / (x[i] - x[j])
        result += term

    return result

def newtons_divided_difference(x, fx, target_x):
    n = len(x)

    h = x[1] - x[0]
    for i in range(1, n - 1):
        if (x[i + 1] - x[i]) == h:
            return "Newton's Divided Difference is not suitable for uniform nodes."

    divided_diff = [[0] * n for _ in range(n)]  # Table of divided differences

    # Fill the first column with fx
    for i in range(n):
        divided_diff[i][0] = fx[i]

    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x[i + j] - x[i])

    # Apply Newton's divided difference formula
    result = divided_diff[0][0]
    term = 1
    for i in range(1, n):
        term *= (target_x - x[i - 1])
        result += term * divided_diff[0][i]

    return result



x = [1, 2, 4, 8]
fx = [1, 8, 27, 64]  # Corresponding to f(x) = x^3
target_x = 3

# Newton's Forward Interpolation
result_forward = newtons_forward_interpolation(x, fx, target_x)
print(f"Newton's Forward Interpolation: {result_forward}")

# Lagrange's Interpolation
result_lagrange = lagranges_interpolation(x, fx, target_x)
print(f"Lagrange's Interpolation: {result_lagrange}")

# Newton's Divided Difference
result_divided = newtons_divided_difference(x, fx, target_x)
print(f"Newton's Divided Difference: {result_divided}")
