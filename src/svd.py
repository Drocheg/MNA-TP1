import numpy as np

def svd(matrix):
    auxMatrix = matrix @ matrix.T
    eigen_values, U = _eig(auxMatrix)
    indexSort = np.argsort(np.absolute(eigen_values))[::-1]
    U = U[indexSort]
    eigen_values = eigen_values[indexSort]
    V = U @ matrix
    col_norms = np.linalg.norm(V, axis=1)
    V = V / col_norms[:, None]
    return eigen_values, V

def gram_schmidt(matrix):
    m, n = matrix.shape
    q = np.zeros((m, n))
    r = np.zeros((n, n))
    for j in range(n):
        v = matrix[:, j]
        for i in range(j):
            r[i, j] = np.matmul(q[:, i].T, matrix[:, j])
            v = v.squeeze() - np.dot(r[i, j], q[:, i])
        r[j, j] = np.linalg.norm(v)
        q[:, j] = np.divide(v, r[j, j]).squeeze()
    return q, r


def _eig(matrix):
    a = np.matrix(matrix, dtype=np.float64)

    q, r = gram_schmidt(a)
    a = np.matmul(r, q)
    s = q

    for i in range(50):
        q, r = gram_schmidt(a)
        a = np.matmul(r, q)
        s = np.matmul(s, q)
        if np.allclose(a, np.diagflat(np.diag(a)), atol=1e-4):
            break

    eigenvalues = np.diag(a)
    return eigenvalues, s


def eig_with_variance_explained(eigen_values, eigen_vectors, variance_percentage):
    total_variance = sum(eigen_values)
    variance_threshold = total_variance * variance_percentage
    partial_sum = 0
    eigen_count = 0
    while partial_sum <= variance_threshold:
        partial_sum += eigen_values[eigen_count]
        eigen_count += 1
    return eigen_count


def variance_explained_from_eigen_values(eigen_values, amount):
    return sum(eigen_values[:amount]) / sum(eigen_values)
