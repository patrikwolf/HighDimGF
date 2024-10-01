###################################################################################################
# Example 2: Exploration of Polynomial Decay
# Python
# Harvard University
###################################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt

import matrix_factorization.definitions_factorization as def_fact
import matrix_factorization.gradient_descent_factorization as gd_fact


def generate_factorization_matrix(n, d1, d2, small_gap=1, large_gap=2):
    # Target
    small_eigs = np.linspace(small_gap, small_gap + 0.9999999, d1)
    large_eigs = np.linspace(small_gap + large_gap + 1, small_gap + large_gap + 1.9999999, d2)
    zeros = np.zeros(n - d1 - d2)

    # Construct matrix
    H = np.diag(np.concatenate((small_eigs, large_eigs, zeros)))
    return H


def plot_test_loss(results, loglog=False):
    # Create figure
    plt.figure(figsize=(9.2, 7))

    # Meta data
    colors = ['#eb3323', '#4d9047']

    # Case 1

    # Extract data
    time_steps = results[0]['time_numerical']
    test_loss = results[0]['num_results']['loss_to_global_opt']

    # Find index where time_steps is greater or equal to 1
    search_one = (time_steps > 1)
    index_one = len(time_steps) - np.sum(search_one) - 1

    if loglog:
        p = plt.loglog(time_steps, test_loss, color=colors[0], label=r'$W_t$')
        plt.loglog(time_steps[index_one:], 0.045 / (time_steps[index_one:] ** 2), '--', color=p[0].get_color(), label=r'$c/t^2$')
    else:
        p = plt.semilogy(time_steps, test_loss, color=colors[0], label=r'$W_t$')
        plt.semilogy(time_steps[index_one:], 0.045 / (time_steps[index_one:] ** 2), '--', color=p[0].get_color(), label=r'$c/t^2$')

    # Case 2

    # Extract data
    time_steps = results[1]['time_numerical']
    test_loss = results[1]['num_results']['loss_to_global_opt']

    # Find index where time_steps is greater or equal to 1
    search_one = (time_steps > 1)
    index_one = len(time_steps) - np.sum(search_one) - 1

    if loglog:
        p = plt.loglog(time_steps, test_loss, color=colors[1], label=r'$W_t$')
        plt.loglog(time_steps[index_one:], 5 / time_steps[index_one:], '--', color=p[0].get_color(), label=r'$c / t$')
    else:
        p = plt.semilogy(time_steps, test_loss, color=colors[1], label=r'$W_t$')
        plt.semilogy(time_steps[index_one:], 5 / time_steps[index_one:], '--', color=p[0].get_color(), label=r'$c / t$')
    
    plt.xlabel('Time [s]')
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f'Test Loss, n = {n}\n\n')
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == '__main__':
    # Fix random seed
    np.random.seed(0)

    # Size of data matrix
    n = 50

    # Bulk size
    d1 = n // 4
    d2 = n // 2
    d = d1 + d2

    # Parameters for gradient flow
    parameters = {
        'threshold': 2 * 1e-12,
        'step_size': 2 * 1e-3,
        'subsampling_factor': 1e2,
        'min_duration': 1,
        'max_duration': 100,
    }

    start = time.time()

    # Random data matrix
    small_gap = 1
    large_gap = 2
    H = generate_factorization_matrix(n, d1, d2, small_gap, large_gap)

    # Estimated rank
    r_values = [(n + d) // 2, d2 // 2]

    # Initialize array
    results = []

    for r in r_values:
        # Compute optimal solution
        H_hat = def_fact.get_optimal_approximation_sym(H, n, r)

        # Initial value
        X_0 = np.random.normal(0, 1 / np.sqrt(n), (n, r))

        # Perform optimization with gradient flow
        gf_results = gd_fact.gradient_descent_factorization(H, X_0, parameters,
                                                              min_duration=parameters['min_duration'],
                                                              max_duration=parameters['max_duration'])

        # Compute loss
        numerical_results = def_fact.compute_loss_factorization(H, H_hat, gf_results['Z_tensor'], n)

        # Store results
        results.append({
            'time_numerical': gf_results['time_steps'],
            'gf_results': gf_results,
            'num_results': numerical_results,
            'parameters': parameters,
            'small_gap': small_gap,
            'large_gap': large_gap,
            'H': H,
            'H_hat': H_hat,
            'n': n,
            'r': r,
        })

    # Plot test loss
    plot_test_loss(results, loglog=False)
    plot_test_loss(results, loglog=True)
