###################################################################################################
# Implementation of Gradient Descent for Asymmetric Matrix Factorization
# Python
# Harvard University
# 16.05.2024
###################################################################################################

import numpy as np


def gradient_descent_factorization(H, U_0, V_0, parameters, min_duration=5, max_duration=30):
    """
    Perform alternating gradient descent for asymmetric matrix factorization

    :param H: target matrix
    :param U_0: initial value for U_t
    :param V_0: initial value for V_t
    :param parameters: dictionary with parameters
    :param min_duration: minimum duration
    :param max_duration: maximum duration
    """
    # Initialization
    U_t = U_0
    V_t = V_0
    Z_t = U_t @ V_t.T
    gradient_U = (Z_t - H) @ V_t
    gradient_V = (Z_t - H).T @ U_t

    # Keep track of approximation in every step
    Z_tensor = [Z_t]

    # Number of steps
    num_steps = int(max_duration // parameters['step_size'])

    # Gradient descent with small step size
    for k in range(num_steps):
        # Check for convergence
        if (np.linalg.norm(parameters['step_size'] * gradient_U) < parameters['threshold']
                and np.linalg.norm(parameters['step_size'] * gradient_V) < parameters['threshold']
                and k * parameters['step_size'] >= min_duration):
            break

        # Update step
        U_t = U_t - parameters['step_size'] * gradient_U
        V_t = V_t - parameters['step_size'] * gradient_V

        # Next gradient
        Z_t = U_t @ V_t.T
        gradient_U = (Z_t - H) @ V_t
        gradient_V = (Z_t - H).T @ U_t

        # Add approximation to history
        if (parameters['subsampling_factor'] == 1) or (k % parameters['subsampling_factor'] == 1):
            Z_tensor.append(Z_t)

    # Create array for time
    time = np.arange(len(Z_tensor)) * parameters['step_size'] * parameters['subsampling_factor']

    # Approximation at convergence
    U_inf = U_t
    V_inf = V_t

    return time, Z_tensor, U_inf, V_inf
