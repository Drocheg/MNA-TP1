import numpy as np
from scipy.linalg import eigh

def svd(matrix):
    eigen_values, U = SU(matrix)
    # S = SS(eigen_values, np.shape(images)[0], np.shape(images)[1])
    # V = SV(S, U, images)
    return np.dot(U, matrix)

def SU(matrix):

    auxMatrix = np.dot(matrix, matrix.T)
    #auxMatrix = matrix @ matrix.T
    #eigen_values, eigen_vectors = np.linalg.eig(auxMatrix)
    #eigen_values, eigen_vectors = eigh(auxMatrix, eigvals = (0,0))
    eigen_values, eigen_vectors = _eig(auxMatrix)

  #  eigen_vectors = eigen_vectors.T

    indexSort = np.argsort(np.absolute(eigen_values))[::-1]

    eigen_values = eigen_values[indexSort]
    eigen_vectors = eigen_vectors[indexSort]

    return eigen_values, eigen_vectors

# def SS(eigen_values, n, p):
#
#     S = np.zeros((n,p))
#
#     for i in range(0, n):
#         S[i, i] = 1/np.sqrt(eigen_values[i])
#
#     return S
#
#
#
# def SV(S, U, matrix):
#
#     V = np.dot(np.dot(matrix.T, U), S)
#     return V



def _householder(matrix):
    """ Calculates the QR decomposition using the Householder method.
            Params:
                matrix (np.matrix): the matrix to decompose.
            Returns:
                q (np.ndarray): The Q orthogonal matrix.
                r (np.ndarray): The R upper triangular matrix.
            Reference:
                http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
            """
    m, n = matrix.shape
    q = np.identity(m)
    r = matrix.copy()
    for i in range(0, m - 1):
        v = r[i:m, i]
        s = -np.sign(v[0]).item()
        norm = np.linalg.norm(v)
        u = (r[i, i] - (norm * s)).item()
        v = np.divide(v, u)
        v[0] = 1
        tm = np.matmul(v, v.T) * (-s * u) / norm
        r[i:, :] = np.subtract(r[i:m, :], np.matmul(tm, r[i:m, :]))
        q[:, i:] = q[:, i:] - np.matmul(q[:, i:], tm)
    return q, r

def _gram_schmidt(matrix):
    """ Calculates the QR decomposition using the Gram Schmidt method.
            Params:
                matrix (np.matrix): the matrix to decompose.
            Returns:
                q (np.ndarray): The Q orthogonal matrix.
                r (np.ndarray): The R upper triangular matrix.
            Reference:
                http://web.mit.edu/18.06/www/Essays/gramschmidtmat.pdf
            """
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


def _eig(matrix, method=_gram_schmidt, iterations=50, tolerance=1e-4):
    a = np.matrix(matrix, dtype=np.float64)

    q, r = method(a)
    a = np.matmul(r, q)
    s = q

    for i in range(iterations):
       # print(i)
        q, r = method(a)
        a = np.matmul(r, q)
        s = np.matmul(s, q)
        if np.allclose(a, np.diagflat(np.diag(a)), atol=tolerance):
            break

    eigenvalues = np.diag(a)
    return eigenvalues, s
