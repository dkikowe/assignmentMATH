import numpy as np
from scipy.linalg import lu

# Task 2: LU Factorization Method for Matrix Inversion
A = np.array([
    [50, 107, 36],
    [35, 54, 20],
    [31, 66, 21]
])

P, L, U = lu(A)

L_inv = np.linalg.inv(L)
U_inv = np.linalg.inv(U)

A_inverse = U_inv @ L_inv
print("Inverse of A using LU Factorization:")
print(A_inverse)

# Task 3: Iterative Method for Refining Inverse
A_task3 = np.array([
    [1, 10, 1],
    [2, 0, 1],
    [3, 3, 2]
])

B_initial = np.array([
    [0.4, 2.4, -1.4],
    [0.14, 0.14, -0.14],
    [-0.85, -3.8, 2.8]
])

I = np.eye(3)
E = A_task3 @ B_initial - I
A_inverse_refined = B_initial @ (I - E + E @ E)
print("\nRefined Inverse of A using Iterative Method:")
print(A_inverse_refined)

# Task 4: Power Method to Find Largest Eigenvalue and Eigenvector
A_task4 = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
])

v = np.array([1, 0, 0])
for _ in range(10):
    v_next = np.dot(A_task4, v)
    v_next = v_next / np.linalg.norm(v_next)
    v = v_next

largest_eigenvalue = np.dot(v.T, np.dot(A_task4, v)) / np.dot(v.T, v)
print("\nLargest Eigenvalue using Power Method:")
print(largest_eigenvalue)
print("Corresponding Eigenvector:")
print(v)

# Task 5: Jacobi's Method to Find All Eigenvalues and Eigenvectors
A_task5 = np.array([
    [1, np.sqrt(2), 2],
    [np.sqrt(2), 3, np.sqrt(2)],
    [2, np.sqrt(2), 1]
])

def jacobi_method(A, tol=1e-10, max_iterations=100):
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_iterations):
        max_off_diag = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_off_diag:
                    max_off_diag = abs(A[i, j])
                    p, q = i, j

        if max_off_diag < tol:
            break

        theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])
        cos, sin = np.cos(theta), np.sin(theta)

        J = np.eye(n)
        J[p, p] = cos
        J[q, q] = cos
        J[p, q] = -sin
        J[q, p] = sin

        A = J.T @ A @ J
        V = V @ J

    eigenvalues = np.diag(A)
    eigenvectors = V
    return eigenvalues, eigenvectors

jacobi_eigenvalues, jacobi_eigenvectors = jacobi_method(A_task5)
print("\nEigenvalues using Jacobi's Method:")
print(jacobi_eigenvalues)
print("Eigenvectors using Jacobi's Method:")
print(jacobi_eigenvectors)
