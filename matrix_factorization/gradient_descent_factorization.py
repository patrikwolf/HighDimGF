###################################################################################################
# Implementation of Gradient Descent for Symmetric Matrix Factorization
# Python
# Harvard University
###################################################################################################

import numpy as np
import scipy as sp


def gradient_descent_factorization(H, X_0, parameters, min_duration=0, max_duration=30):
    """
    Perform gradient flow for low-rank matrix approximation

    :param H: target matrix
    :param X_0: initial value
    :param parameters: dictionary with parameters
    :param min_duration: minimum duration
    :param max_duration: maximum duration
    """
    # Initialization
    X_t = X_0
    Z_t = X_t @ X_t.T
    gradient = (Z_t - H) @ X_t

    # Keep track of approximation in every step
    Z_tensor = [Z_t]
    gradient_tensor = [gradient]

    # Number of steps
    num_steps = int(max_duration // parameters['step_size'])

    # Gradient descent with small step size
    for k in range(num_steps):
        # Check for convergence
        if (np.linalg.norm(parameters['step_size'] * gradient) < parameters['threshold']
                and k * parameters['step_size'] >= min_duration):
            break

        # Update step
        X_t = X_t - parameters['step_size'] * gradient

        # Next gradient
        Z_t = X_t @ X_t.T
        gradient = (Z_t - H) @ X_t

        # Add approximation to history
        if (parameters['subsampling_factor'] == 1) or (k % parameters['subsampling_factor'] == 1):
            Z_tensor.append(Z_t)
            gradient_tensor.append(gradient)

    # Create array for time
    time_steps = np.arange(len(Z_tensor)) * parameters['step_size'] * parameters['subsampling_factor']

    # Approximation at convergence
    X_inf = X_t

    # Wrap arrays in dictionary
    results = {
        'time_steps': time_steps,
        'Z_tensor': Z_tensor,
        'X_inf': X_inf,
        'gradient_tensor': gradient_tensor
    }

    return results


def get_analytical_solution(H, X_0, m, time_steps):
    """
    Compute analytical solution for matrix factorization

    :param H: target matrix
    :param X_0: initial value
    :param m: number of columns of X_t
    :param time_steps: time steps from numerical results
    """
    Z_analytical = []

    for idx, t in enumerate(time_steps):
        # Compute Z_t
        try:
            Z_t, A_t, mat_inverse, exp_mat = get_Z_t(H, X_0, t, m)
            Z_analytical.append(Z_t)
        except np.linalg.LinAlgError:
            time_steps = time_steps[:idx]
            break

    # Get (possibly) truncated time steps
    time_steps = time_steps[:len(Z_analytical)]

    return time_steps, Z_analytical


def get_det_equiv(H, H_hat, n, m, time_steps, sigma_X_0, subsampling=40):
    """
    Compute deterministic equivalent for matrix factorization

    :param H: target matrix
    :param H_hat: optimal approximation
    :param n: number of rows of X_t
    :param m: number of columns of X_t
    :param time_steps: time steps from numerical results
    :param sigma_X_0: standard deviation of initial value
    :param subsampling: subsampling factor
    """
    mu_array = []
    s_hat_array = []
    p_hat_array = []
    Z_hat_array = []

    for idx, t in enumerate(time_steps):
        if sigma_X_0 == 0:
            raise ValueError('Standard deviation of initial value must be positive.')

        # Compute A_t and exponential
        A_t = 2 * sp.integrate.quad_vec(lambda s: sp.linalg.expm(2 * s * H), 0, t)[0]
        exp_mat = sp.linalg.expm(t * H)

        # Compute mu_t
        if t > 1e-4:
            prev_mu_t = mu_array[-1]
        else:
            prev_mu_t = m / n
        try:
            mu_t = get_mu_t_fp(n, m, t, prev_mu_t, A_t, sigma_X_0)
            mu_array.append(mu_t)
        except ValueError:
            print(f'---> ValueError: Could not find positive solution mu_t for FP equation at time {t}')
            time_steps = time_steps[:idx]
            break

        # Compute Z_hat_t
        Z_hat_t, M_hat = get_Z_hat_t(A_t, mu_t, exp_mat, n)
        Z_hat_array.append(Z_hat_t)

        # Compute s_hat_t
        s_hat = get_s_hat_value(H_hat, Z_hat_t, n)
        s_hat_array.append(s_hat)

        # Compute p_hat_t
        p_hat = get_p_hat_value(A_t, mu_t, exp_mat, n, sigma_X_0)
        p_hat_array.append(p_hat)

    # Subsampling (if we subsample before the for-loop, we cannot find the fixed point mu_t)
    time_sub = time_steps[::subsampling]
    mu_array = mu_array[::subsampling]
    s_hat_array = s_hat_array[::subsampling]
    p_hat_array = p_hat_array[::subsampling]
    Z_hat_array = Z_hat_array[::subsampling]

    # Add values to results dictionary
    det_equiv = {
        'mu_array': mu_array,
        's_hat_array': s_hat_array,
        'p_hat_array': p_hat_array,
        'Z_hat_array': Z_hat_array
    }

    return time_sub, det_equiv


def get_Z_t(H, X_0, t, m):
    # Integral
    A_t = 2 * sp.integrate.quad_vec(lambda s: sp.linalg.expm(2 * s * H), 0, t)[0]

    # Compute inverse matrix (if possible)
    temp = np.eye(m) + X_0.T @ A_t @ X_0
    cond_number = np.linalg.cond(temp)
    if cond_number > 1e14:
        raise np.linalg.LinAlgError
    mat_inverse = np.linalg.solve(temp, np.eye(m))

    # Compute pre-factor
    exp_mat = sp.linalg.expm(t * H)
    W = exp_mat @ X_0

    # Compute analytical Solution
    Z_t = W @ mat_inverse @ W.T

    return Z_t, A_t, mat_inverse, exp_mat


def mu_t_fp_eq(mu_t, n, m, A_t, sigma_X_0):
    psi = m/n
    return (sigma_X_0 ** 2) * ((1 / n) * np.trace(np.linalg.inv(mu_t * A_t + np.eye(n))) - (1 - psi))


def scaled_mu_t_zero_eq(beta, n, m, scaling, A_t, sigma_X_0):
    psi = m / n
    rhs = (sigma_X_0 ** 2) * ((1 / n) * np.trace(np.linalg.inv(beta * scaling * A_t + np.eye(n))) - (1 - psi))
    return beta * scaling - rhs


def get_mu_t_fp(n, m, t, prev_mu_t, A_t, sigma_X_0):
    psi = m/n
    if t < 1e-4:
        result = sp.optimize.fixed_point(mu_t_fp_eq, [psi], args=(n, m, A_t, sigma_X_0))
        mu_t = result[0]
    else:
        results = sp.optimize.fsolve(scaled_mu_t_zero_eq, np.array([1]), args=(n, m, prev_mu_t, A_t, sigma_X_0), xtol=1e-10)
        mu_t = results[0] * prev_mu_t
        if results[0] < 0:
            raise ValueError(f'Negative mu_t: {mu_t} at time {t}')
    return mu_t


def get_Z_hat_t(A_t, mu_t, exp_mat, n):
    xi = 1 / mu_t
    M_hat = np.linalg.inv((xi * np.eye(n) + A_t))
    Z_hat_t = exp_mat @ M_hat @ exp_mat
    return Z_hat_t, M_hat


def get_s_hat_value(H_hat, Z_hat_t, n):
    s_hat = (1 / n) * np.trace(H_hat @ Z_hat_t)
    return s_hat


def get_p_hat_value(A_t, mu_t, exp_mat, n, sigma_X_0):
    exp_squared = exp_mat @ exp_mat
    inverse = np.linalg.inv(mu_t * A_t + np.eye(n))
    inverse_square = inverse @ inverse
    first_summand = (mu_t ** 2) * (1 / n) * np.trace(exp_squared @ exp_squared @ inverse_square)
    den = 1 + (sigma_X_0 ** 2) / n * np.trace(A_t @ inverse_square)
    trace = (1 / n) * np.trace(exp_squared @ inverse_square)
    second_summand = (sigma_X_0 ** 2) * mu_t / den * (trace ** 2)
    return first_summand + second_summand
