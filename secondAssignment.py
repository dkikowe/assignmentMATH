import numpy as np

tolerance = 1e-10
A = np.array([[3, -5, 47, 20],
              [11, 16, 17, 10],
              [56, 22, 11, -18],
              [17, 66, -12, 7]])

b = np.array([18, 26, 34, 82])

def Cramer(A, b):
    n = len(A)
    det_A = np.linalg.det(A)
    if det_A == 0:
        print("Determinant is zero. Cramer's Rule is not applicable.")
        return
    X = np.zeros(n)
    for i in range(n):
        A_copy = A.copy()
        A_copy[:, i] = b
        X[i] = np.linalg.det(A_copy) / det_A
    return X


def Gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        max_row = np.argmax(abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        for j in range(i + 1, n):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]

    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, :-1], X)

    return X


def Jacobi(A, b, tolerance):
    max_iterations = 1000
    n = len(A)
    X = np.zeros(n)
    for _ in range(max_iterations):
        X_new = np.zeros_like(X)
        for i in range(n):
            sum_ = 0
            for j in range(n):
                if i != j:
                    sum_ += A[i, j] * X[j]
            X_new[i] = (b[i] - sum_) / A[i, i]

        if np.linalg.norm(X_new - X, ord=np.inf) < tolerance:
            return X_new

        X = X_new

    print("Jacobi method did not converge within the maximum number of iterations")
    return


def Gauss_seidel(A, b, tolerance):
    max_iterations = 1000
    n = len(A)
    X = np.zeros(n)
    for _ in range(max_iterations):
        X_new = X.copy()
        for i in range(n):
            sum_ = 0
            for j in range(n):
                if i != j:
                    sum_ += A[i, j] * X_new[j]
            X_new[i] = (b[i] - sum_) / A[i, i]

        if np.linalg.norm(X_new - X, ord=np.inf) < tolerance:
            return X_new

        X = X_new

    print("Gauss-Seidel method did not converge within the maximum number of iterations")
    return


def reorder_to_diagonal_dominance(A, b):
    n = len(A)
    for i in range(n):
        row_max = max(range(i, n), key=lambda r: abs(A[r, i]))
        A[[i, row_max]] = A[[row_max, i]]
        b[[i, row_max]] = b[[row_max, i]]
    return A, b




A_dd, b_dd = reorder_to_diagonal_dominance(A.copy(), b.copy())

cramer_solution = Cramer(A, b)
gaussian_solution = Gaussian_elimination(A, b)
jacobi_solution = Jacobi(A_dd, b_dd, tolerance)
gauss_seidel_solution = Gauss_seidel(A_dd, b_dd, tolerance)

print("Cramer's Rule Solution:", cramer_solution)
print("Gaussian Elimination Solution:", gaussian_solution)
print("Jacobi Method Solution:", jacobi_solution)
print("Gauss-Seidel Method Solution:", gauss_seidel_solution)