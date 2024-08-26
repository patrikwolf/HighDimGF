###################################################################################################
# Helper Functions to Plot Results
# Python
# Harvard University
###################################################################################################

import os
import sys
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utilities.time as time_helper

if sys.platform == "darwin":
    # Interactive plots on macOS
    mpl.use('macosx')

if shutil.which('latex'):
    # Plot settings (LaTeX)
    plt.rcParams.update({'text.usetex': True,
                         'font.family': 'serif',
                         'text.latex.preamble': r'\usepackage{amsfonts}'})


def plot_results(results, parameters, save=False, det_equiv=True):
    fig = plt.figure(layout='constrained', figsize=(14, 22))
    fig.suptitle(parameters['title'], fontsize=16)
    subfigs = fig.subfigures(5, 1, hspace=0.08)

    # First row
    first_row = subfigs[0].subfigures(1, 3, wspace=0.1)

    # Display parameters
    info = (r'\textbf{Notation}' + '\n\n' +
            r'$H\in\mathbb{R}^{n\times n}$' + '\n' +
            r'$X_t\in\mathbb{R}^{n\times m}$' + '\n' +
            r'$Z_t = X_tX_t^T\in\mathbb{R}^{n\times n}$' + '\n\n\n' +
            r'\textbf{Parameters}' + '\n\n' +
            f'Dimension: $n = {results["n"]}$\n'
            f'Rank of $H$: $d = {results["d"]}$\n'
            f'Estimated rank: $m = {results["m"]}$\n\n\n'
            r'\textbf{Initialization}' + '\n\n' +
            '$\mathbb{E}[(X_0)_{ij}] = 0$\n'
            r'$\mathrm{Var}[(X_0)_{ij}] = ' + f'{parameters["sigma_X_0"] ** 2:.2f} / n$\n\n\n' +
            r'\textbf{Gradient Flow Parameters}' + '\n\n' +
            r'Step size = $10^{' + '{:.2f}'.format(np.log10(parameters['step_size'])) + '}$')

    ax = first_row[0].subplots(1, 1)
    ax.axis('off')
    ax.text(0.55, 0.6, info,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

    # Plot eigenvalue distribution
    m = results['m']
    eigenvalues = np.linalg.eigvals(results['H'])
    cut_off = plot_eigenvalue_distribution(first_row[1], eigenvalues, m)

    # Plot non-zero eigenvalues
    non_zero_eigenvalues = eigenvalues[abs(eigenvalues) > 1e-6]
    plot_eigenvalue_distribution(first_row[2], non_zero_eigenvalues, m, non_zero=True, cut_off=cut_off)

    # Second row
    sec_row = subfigs[1].subfigures(1, 2, wspace=0.1)

    # Loss to target
    title = r'Loss to Target: $\frac{1}{n}\,\Vert H - Z_t\Vert_\mathrm{F}^2$'
    attribute = 'loss_to_target'
    add_subplot_linear(sec_row[0], title, results, attribute)

    # Loss to target minus limiting value
    title = r'Loss to Target minus Limit: $\frac{1}{n}\,\Vert H - Z_t\Vert_\mathrm{F}^2 - \frac{1}{n}\,\Vert H - \hat{H}\Vert_\mathrm{F}^2$'
    attribute = 'loss_to_target'
    limit = results['numerical_results']['optimal_loss']
    add_subplot_logarithmic(sec_row[1], title, results, attribute, limit)

    # Third row
    third_row = subfigs[2].subfigures(1, 2, wspace=0.1)

    # Loss to optimal solution in linear scale
    title = r'Loss to Optimal Approximation: $\frac{1}{n}\,\Vert\hat{H} - Z_t\Vert_\mathrm{F}^2$'
    attribute = 'loss_to_global_opt'
    add_subplot_linear(third_row[0], title, results, attribute, det_equiv=det_equiv)

    # Loss to optimal solution in logarithmic scale
    title = r'Loss to Optimal Approximation: $\frac{1}{n}\,\Vert\hat{H} - Z_t\Vert_\mathrm{F}^2$'
    attribute = 'loss_to_global_opt'
    add_subplot_logarithmic(third_row[1], title, results, attribute, det_equiv=det_equiv)

    # Fourth row
    fourth_row = subfigs[3].subfigures(1, 2, wspace=0.1)

    # Commutator
    title = r'Squared Norm of Commutator: $\Vert H Z_t - Z_t H\Vert_\mathrm{F}^2$'
    attribute = 'norm_commutator'
    add_subplot_linear(fourth_row[0], title, results, attribute)

    # Fifth row
    fifth_row = subfigs[4].subfigures(1, 2, wspace=0.1)

    # Order parameter s_t
    title = r'Order Parameter: $s_t = \frac{1}{n}\,\mathrm{tr}(\hat{H}Z_t)$'
    attribute = 's_array'
    attribute_hat = 's_hat_array'
    add_subplot_det_equiv(fifth_row[0], title, results, attribute, attribute_hat)

    # Order parameter p_t
    title = r'Order Parameter: $p_t = \frac{1}{n}\,\Vert Z_t\Vert_\mathrm{F}^2$'
    attribute = 'p_array'
    attribute_hat = 'p_hat_array'
    add_subplot_det_equiv(fifth_row[1], title, results, attribute, attribute_hat)

    # Save
    if save:
        if not os.path.isdir('../graphics'):
            os.mkdir('../graphics')
        timestamp = time_helper.get_dash_timestamp()
        plt.savefig('../graphics/' + parameters['file_name'] + timestamp + '.pdf',
                    format='pdf')

    # Show figure
    plt.show()


def plot_eigenvalue_distribution(fig, eigenvalues, m, non_zero=False, cut_off=None):
    # Title
    if non_zero:
        title = 'Distribution of Non-Zero Eigenvalues'
        num_bins = 40
    else:
        title = 'Eigenvalue Distribution of Target Matrix'
        num_bins = 80
    fig.suptitle(title)

    # Subplots
    ax = fig.subplots(1, 1)

    # Sorted eigenvalues
    sorted_ev = np.sort(eigenvalues)[::-1]

    # Determine borders
    left_border = sorted_ev[-1] - 0.5
    right_border = sorted_ev[0] + 0.5

    # Bins
    bins = np.linspace(left_border, right_border, num_bins)

    # Plot histogram for eigenvalues
    ax.hist(eigenvalues,
            bins=bins,
            density=True,
            label='Eigenvalues of Target')

    # Show cut-off
    if cut_off is not None:
        ev_cut_off = cut_off
    else:
        ev_cut_off = sorted_ev[m - 1]
    ax.axvline(x=ev_cut_off, label=r'Cut-off', color='m', linestyle=(0, (6, 10)))

    # Formatting
    ax.grid()
    ax.legend()
    return ev_cut_off


def add_subplot_linear(fig, title, results, attribute, det_equiv=False):
    # Title
    fig.suptitle(title)

    # Subplots
    ax = fig.subplots(1, 1)

    # Plot results
    ax.plot(results['time_numerical'], results['numerical_results'][attribute],
            label='Gradient Flow')
    ax.plot(results['time_analytical'], results['analytical_results'][attribute], '--',
            label='Analytical Solution')
    bottom, top = ax.get_ylim()
    if det_equiv:
        loss_det_equiv = (np.array(results['det_equiv']['p_hat_array']) -
                          2 * np.array(results['det_equiv']['s_hat_array']) +
                          results['numerical_results']['h_value'])
        ax.plot(results['time_det_equiv'], loss_det_equiv, '*', label='Deterministic Equivalent')
        new_bottom = min(bottom, max(0.5 * top, 0.9 * min(loss_det_equiv)))
        new_top = max(top, min(1.5 * top, 1.1 * max(loss_det_equiv)))
        ax.set_ylim(new_bottom, new_top)

    # Formatting
    ax.set_xlabel('Time [s]')
    ax.grid()
    ax.legend()
    ax.margins(x=0)


def add_subplot_logarithmic(fig, title, results, attribute, limit=0, det_equiv=False):
    # Title
    fig.suptitle(title)

    # Subplots
    ax = fig.subplots(1, 1)

    # Plot results
    ax.semilogy(results['time_numerical'], results['numerical_results'][attribute] - limit,
                label='Gradient Flow')
    ax.semilogy(results['time_analytical'], results['analytical_results'][attribute] - limit, '--',
                label='Analytical Solution')
    bottom, top = ax.get_ylim()
    if det_equiv:
        loss_det_equiv = (np.array(results['det_equiv']['p_hat_array']) -
                          2 * np.array(results['det_equiv']['s_hat_array']) +
                          results['numerical_results']['h_value'])
        ax.semilogy(results['time_det_equiv'], loss_det_equiv, '*', label='Deterministic Equivalent')
        new_bottom = min(bottom, max(0.5 * top, 0.9 * min(loss_det_equiv)))
        new_top = max(top, min(1.5 * top, 1.1 * max(loss_det_equiv)))
        ax.set_ylim(new_bottom, new_top)

    # Formatting
    ax.set_xlabel('Time [s]')
    ax.grid()
    ax.legend()
    ax.margins(x=0)


def add_subplot_det_equiv(fig, title, results, attribute, attribute_hat):
    # Title
    fig.suptitle(title)

    # Subplots
    ax = fig.subplots(1, 1)

    # Plot results
    ax.plot(results['time_numerical'], results['numerical_results'][attribute],
            label='Gradient Flow')
    ax.plot(results['time_analytical'], results['analytical_results'][attribute], '--',
            label='Analytical Solution')
    bottom, top = ax.get_ylim()
    ax.plot(results['time_det_equiv'], results['det_equiv'][attribute_hat], '*',
            label='Deterministic Equivalent')
    ax.set_ylim(bottom, max(top, min(1.5 * top, 1.1 * max(results['det_equiv'][attribute_hat]))))

    # Formatting
    ax.set_xlabel('Time [s]')
    ax.grid()
    ax.legend()
    ax.margins(x=0)
