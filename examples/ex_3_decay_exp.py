###################################################################################################
# Example 3: Exploration of Exponential Decay
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


def plot_test_loss(results):
    # Create figure
    plt.figure(figsize=(9.2, 7))

    # Meta data
    colors = ['#deaf3c', '#ec5281']
    exp_labels = [r'$c\cdot\exp(-t)$', r'$c\cdot\exp(-2t)$']
    exp_factor = [0.4, 12]
    
    for k in range(2):
        # Extract data
        time_steps = results[k]['time_numerical']
        test_loss = results[k]['num_results']['loss_to_global_opt']

        # Plots
        plt.semilogy(time_steps, test_loss, color=colors[k], label=r'$W_t$')
        plt.semilogy(time_steps, exp_factor[k] * np.exp(-(k+1) * time_steps), '--', color=colors[k],
                 label=exp_labels[k])
    
    plt.xlabel('Time [s]')
    plt.grid()
    plt.legend()
    plt.title(f'Test Loss, n = {n}\n\n')
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == '__main__':
    # Fix random seed
    np.random.seed(0)

    # Size of data matrix
    n = 100

    # Bulk size
    d1 = n // 4
    d2 = n // 2

    # Parameters for gradient flow
    parameters = {
        'threshold': 2 * 1e-12,
        'step_size': 2 * 1e-3,
        'subsampling_factor': 1e1,
        'min_duration': 1,
        'max_duration': 6,
    }

    # Random data matrix
    small_gap = 1
    large_gap = 2
    H = generate_factorization_matrix(n, d1, d2, small_gap, large_gap)

    # Estimated rank
    r_values = [d1 + d2, d2]

    # Initialize array
    results = []

    for r in r_values:
        # Compute optimal solution
        H_hat = def_fact.get_optimal_approximation_sym(H, n, r)

        # Initial value
        X_0 = np.random.normal(0, 1 / np.sqrt(n), (n, r))

        # Perform optimization with gradient flow
        start = time.time()
        gf_results = gd_fact.gradient_descent_factorization(H, X_0, parameters,
                                                        min_duration=parameters['min_duration'],
                                                        max_duration=parameters['max_duration'])
        end = time.time()
        print(f'--> Elapsed time for gradient descent: {(end - start):.2f} seconds')

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

    # Plot test loss (exponential terms)
    plot_test_loss(results)
