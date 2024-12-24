import numpy as np


A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])
B = np.array([18, 26, 34, 82])


def cramers_rule(A, B):
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        raise ValueError("The system has no unique solution (det(A) = 0)")

    n = len(B)
    solutions = []
    for i in range(n):
        A_temp = A.copy()
        A_temp[:, i] = B
        solutions.append(np.linalg.det(A_temp) / det_A)

    return np.array(solutions)


def gaussian_elimination(A, B):
    n = len(B)
    A_aug = np.hstack([A, B.reshape(-1, 1)])

    for i in range(n):
        max_row = np.argmax(np.abs(A_aug[i:, i])) + i
        A_aug[[i, max_row]] = A_aug[[max_row, i]]

        A_aug[i] = A_aug[i] / A_aug[i, i]
        for j in range(i + 1, n):
            A_aug[j] -= A_aug[j, i] * A_aug[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = A_aug[i, -1] - np.dot(A_aug[i, i + 1:n], x[i + 1:n])

    return x

def jacobi_method(A, B, iterations=100, tol=1e-10):
    n = len(B)
    x = np.zeros(n)
    x_new = np.zeros(n)

    for _ in range(iterations):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (B[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new.copy()

    return x

def gauss_seidel(A, B, iterations=100, tol=1e-10):
    n = len(B)
    x = np.zeros(n)

    for _ in range(iterations):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (B[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new.copy()

    return x

cramer_solution = cramers_rule(A, B)
gauss_solution = gaussian_elimination(A, B)
jacobi_solution = jacobi_method(A, B)
gauss_seidel_solution = gauss_seidel(A, B)


print("cramer's solution:", cramer_solution)
print("gaussian elimination olution:", gauss_solution)
print("jacobi method solution:", jacobi_solution)
print("Gauss-Seidel method solution:", gauss_seidel_solution)
