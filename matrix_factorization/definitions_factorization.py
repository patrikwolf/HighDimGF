###################################################################################################
# Definitions of Central Quantities and Parameters for Matrix Factorization
# Python
# Harvard University
###################################################################################################

import numpy as np
from scipy.stats import ortho_group


def generate_data_matrix(n, d, structure='general', additional_dim=0):
    """
    Generate well-conditioned random data matrix with of dimension n x n and rank d.

    :param n: Dimension of square matrices
    :param d: Rank of matrix
    :param structure: Type of matrix (general symmetric matrix, diagonal, psd)
    """
    if structure == 'general':
        U = ortho_group.rvs(dim=n)
        V = ortho_group.rvs(dim=additional_dim)
        singular_values = np.concatenate((np.random.uniform(low=2, high=2.25, size=d), np.zeros(n - d)))
        D = np.zeros((n, additional_dim))
        np.fill_diagonal(D, singular_values)
        return U @ D @ V.T
    if structure == 'symmetric':
        U = ortho_group.rvs(dim=n)
        D = np.diag(np.concatenate((np.random.uniform(low=2, high=2.25, size=d // 2),
                                    np.random.uniform(low=-2.25, high=-2, size=d - d // 2),
                                    np.zeros(n - d))))
        return U @ D @ U.T
    elif structure == 'diagonal':
        D = np.diag(np.concatenate((np.random.uniform(low=2, high=2.25, size=d // 2),
                                    np.random.uniform(low=-2.25, high=-2, size=d - d//2),
                                    np.zeros(n - d))))
        return D
    elif structure == 'psd':
        D = np.diag(np.concatenate((np.random.uniform(low=0.5, high=0.75, size=d),
                                    np.zeros(n - d))))
        return D
    else:
        raise ValueError('Invalid structure type.')


def get_optimal_approximation_sym(H, n, m):
    """
    Determine optimal approximation of H with product XX^T.

    :param H: Symmetric target matrix
    :param n: Dimension of square matrices
    :param m: Number of columns of X
    """

    # Check if target matrix is symmetric
    if not np.allclose(H, H.T):
        raise ValueError('Target matrix is not symmetric.')

    # Eigendecomposition for symmetric matrix (EVs are sorted in ascending order and real)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Set negative eigenvalues to zero
    eigenvalues = np.array([max(0, ev) for ev in eigenvalues])

    # Retain leading min(n, m) eigenvalues
    k = min(n, m)
    eigenvalues[:-k] = 0

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def get_optimal_approximation_asym(H, n, l, m):
    """
    Determine optimal approximation of H with a product of UV^T.

    :param H: Target matrix
    :param n: First dimension of target
    :param l: Second dimension of target
    :param m: Number of columns of U and V
    """

    # Singular value decomposition
    U, S, Vh = np.linalg.svd(H)

    # Truncate singular values
    num_eigs = min(n, m)
    truncated_sv = S[:num_eigs]
    S_truncated = np.zeros((n, l))
    np.fill_diagonal(S_truncated[:num_eigs, :num_eigs], truncated_sv)

    return U @ S_truncated @ Vh


def compute_loss_factorization(H, H_hat, Z_tensor, n):
    """
    Compute loss for factorization problem

    :param H: Target matrix
    :param H_hat: Optimal approximation of H with rank-m matrix
    :param Z_tensor: Tensor of approximations at time steps t
    :param n: Dimension of square matrices
    """

    # Initialize arrays
    loss_to_target = np.zeros(len(Z_tensor))
    loss_to_global_opt = np.zeros(len(Z_tensor))
    s_array = np.zeros(len(Z_tensor))
    p_array = np.zeros(len(Z_tensor))

    # Compute loss and order parameters
    for k in range(len(Z_tensor)):
        loss_to_target[k] = get_loss_to_target(H, Z_tensor[k], n)
        loss_to_global_opt[k] = get_loss_to_global_opt(H_hat, Z_tensor[k], n)
        s_array[k] = get_s_value(H_hat, Z_tensor[k], n)
        p_array[k] = get_p_value(Z_tensor[k], n)

    h = get_h_value(H_hat, n)
    optimal_loss = get_loss_to_target(H, H_hat, n)

    results = {
        'loss_to_target': loss_to_target,
        'loss_to_global_opt': loss_to_global_opt,
        's_array': s_array,
        'p_array': p_array,
        'h_value': h,
        'optimal_loss': optimal_loss,
    }

    return results


def get_loss_to_target(H, Z_t, n):
    return (1 / n) * np.linalg.norm(H - Z_t, ord='fro') ** 2


def get_loss_to_global_opt(H_hat, Z_t, n):
    return (1 / n) * np.linalg.norm(H_hat - Z_t, ord='fro') ** 2


def get_s_value(H_hat, Z_t, n):
    return (1 / n) * np.trace(Z_t.T @ H_hat)


def get_p_value(Z_t, n):
    return (1 / n) * np.linalg.norm(Z_t, ord='fro') ** 2


def get_h_value(H_hat, n):
    return (1 / n) * np.linalg.norm(H_hat, ord='fro') ** 2


def get_convergence_error(Z_inf, Z_t, n):
    return (1 / n) * np.linalg.norm(Z_inf - Z_t, ord='fro') ** 2
