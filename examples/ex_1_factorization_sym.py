###################################################################################################
# Example 1: Plain Vanilla Matrix Factorization
# Python
# Harvard University
###################################################################################################

import time
import numpy as np

import matrix_factroization.definitions_factorization as def_fact
import matrix_factroization.gradient_descent_factorization as gd_fact
import matrix_factroization.plot_helper as plot_helper

if __name__ == '__main__':
    # Timer
    start = time.time()

    # Fix random seed
    np.random.seed(0)

    # Size of data matrix
    n = 100

    # Rank of target matrix
    d = 75

    # Estimated rank
    m = 50

    # Parameters for gradient flow
    parameters = {
        'title': '\n' + r'\textbf{Example 1: Symmetric Matrix Factorization}' + '\n',
        'file_name': 'ex_1_matrix_fact_',
        'threshold': 2 * 1e-6,
        'step_size': 2 * 1e-3,
        'subsampling_factor': 1e1,
        'min_duration': 6,
        'max_duration': 80,
        'sigma_X_0': 1,
    }

    # Random data matrix
    H = def_fact.generate_data_matrix(n, d, structure='psd')

    # Optimal approximation
    H_hat = def_fact.get_optimal_approximation_sym(H, n, m)

    # Initial value
    X_0 = parameters['sigma_X_0'] * np.random.normal(0, 1 / np.sqrt(n), (n, m))

    # Perform optimization with gradient flow
    print(f'Started gradient descent after {time.time() - start} seconds.')
    gf_results = gd_fact.gradient_descent_factorization(H, X_0, parameters,
                                                        min_duration=parameters['min_duration'],
                                                        max_duration=parameters['max_duration'])
    print(f'Finished gradient descent after {time.time() - start} seconds.')

    # Get analytical solution
    print(f'Started computation of analytical solution after {time.time() - start} seconds.')
    final_time = gf_results['time_steps'][-1]
    detailed = final_time // 3
    time_steps = np.concatenate((np.linspace(0, detailed, 20), np.linspace(detailed, final_time, 20)))
    time_analytical, Z_analytical = gd_fact.get_analytical_solution(H, X_0, m, time_steps)
    print(f'Finished computation of analytical solution after {time.time() - start} seconds.')

    # Compute deterministic equivalent
    print(f'Started computation of deterministic equivalent after {time.time() - start} seconds.')
    time_det_equiv, det_equiv = gd_fact.get_det_equiv(H, H_hat, n, m, gf_results['time_steps'],
                                                      parameters['sigma_X_0'],
                                                      subsampling=70)
    print(f'Finished computation of deterministic equivalent after {time.time() - start} seconds.')

    # Compute loss
    numerical_results = def_fact.compute_loss_factorization(H, H_hat, gf_results['Z_tensor'], n)
    analytical_results = def_fact.compute_loss_factorization(H, H_hat, Z_analytical, n)

    # Compute norm of commutator
    Z_numerical = gf_results['Z_tensor']
    numerical_results['norm_commutator'] = (1 / n) * np.linalg.norm(H @ Z_numerical - Z_numerical @ H,
                                                                    axis=(1, 2), ord='fro') ** 2
    analytical_results['norm_commutator'] = (1 / n) * np.linalg.norm(H @ Z_analytical - Z_analytical @ H,
                                                                     axis=(1, 2), ord='fro') ** 2

    # Store results in dictionary
    results_factorization = {
        'time_numerical': gf_results['time_steps'],
        'gradient_tensor': gf_results['gradient_tensor'],
        'numerical_results': numerical_results,
        'time_analytical': time_analytical,
        'analytical_results': analytical_results,
        'time_det_equiv': time_det_equiv,
        'det_equiv': det_equiv,
        'parameters': parameters,
        'H': H,
        'H_hat': H_hat,
        'n': n,
        'd': d,
        'm': m,
    }

    # Plot results
    plot_helper.plot_results(results_factorization, parameters, save=True)

    # Total elapsed time
    end = time.time()
    print(f'--> n = {n}')
    print(f'--> Elapsed time for execution of complete script: {(end - start):.2f} '
          f'seconds which corresponds to roughly {(end - start) / 60:.2f} minutes.')
