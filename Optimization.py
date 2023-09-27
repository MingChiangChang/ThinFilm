# This file includes the optimization process.

from scipy.optimize import least_squares
from DataProcessing import data_smoothing
from scipy.interpolate import CubicSpline
import numpy as np
from DataProcessing import plot_reflectance, plot_nk
import time

iteration_counter = [0]


def optimize_nk(multilayer, layer_index, data, n_points, weight_n=1, weight_k=1, weight_second_diff_n=1, weight_second_diff_k=1, ftol=10e-4, smooth=False, initial_reflectance=None, reflectance_plot_path='', nk_plot_path='', start_time=None):
    """
    Optimize the control points of n, k, and the thickness for a specific layer in the multilayer using Levenberg-Marquardt algorithm.

    Parameters:
    - multilayer (ThinFilmSystem): The multilayer system object.
    - layer_index (int): The index of the layer in the multilayer system to be optimized.
    - data (pd.DataFrame): Experimental data where `data['wavelength']` is the wavelength and `data['reflectance']` is the experimental reflectance.
    - n_points (int): Number of points used to fit n, k.
    - weight_n1: Weight of the sum of the difference between two adjacent ns. Higher the weight, emphasize more on the smoothing of n.
    - weight_k1: Weight of the sum of the difference between two adjacent ks. Higher the weight, emphasize more on the smoothing of k.
    - weight_second_diff_n: Weight of the sum of the second order difference between adjacent ns.
    - weight_second_diff_k: Weight of the sum of the second order difference between adjacent ks.

    Returns:
    - optimized_params (list of float): A list containing the optimized values [n1, n2, ..., k1, k2, ..., thickness].
    - R_optimized (list of float): The reflectance calculated using the optimized parameters.
    """

    # Define the residual function
    def residuals(params):
        """
        Calculate the residuals between calculated reflectance and experimental reflectance.

        Parameters:
        - params (list of float): List of parameters [n1, n2, ..., k1, k2, ..., thickness] to be optimized.
        Returns:
        - list of float: Residuals = calculated_reflectance - experimental_reflectance.
        """
        # The following codes are used to plot R and nk every 'frequency' number (10, 50, etc.) of iterations, to show the process of optimization
        # _______________________________________________________________________________________________________________________________________________
        # Calculate the elapsed time from the start of the optimization
        elapsed_time = time.time() - start_time if start_time else 0

        # Check if initial_reflectance is provided and valid
        if initial_reflectance is not None and initial_reflectance.size > 0:
            iteration_counter[0] += 1
            frequency = 50  # Modify the frequency as needed
            if iteration_counter[0] % frequency == 0:

                # Calculate the current reflectance (optimal during optimization)
                current_reflectance, _, _ = multilayer.calculate_RTA(
                    data['wavelength'])

                # Plot reflectance with your settings
                reflectance_filename = f"reflectance_plot_{iteration_counter[0]}.png"
                plot_reflectance(data, multilayer=multilayer, layer_index=layer_index, calculated_data=initial_reflectance,
                                 optimal_data=current_reflectance, save=True, show=False, filepath=reflectance_plot_path, filename=reflectance_filename, iteration=iteration_counter[0], elapsed_time=elapsed_time)

                # Plot n and k with your settings
                nk_filename = f"nk_plot_{iteration_counter[0]}.png"
                plot_nk(data, params, n_points, multilayer, layer_index=layer_index,
                        save=True, show=False, filepath=nk_plot_path, filename=nk_filename, iteration=iteration_counter[0], elapsed_time=elapsed_time)
        # _______________________________________________________________________________________________________________________________________________

        # Extract spline control points and thickness from params
        n_control = params[: n_points]
        k_control = params[n_points: 2 * n_points]
        thickness = params[-1]

        # Put n_control, k_control, and thickness into the multilayer system
        wavelength = multilayer.layers[layer_index].wavelength
        original_n_spline_cubic = CubicSpline(wavelength, n_control)
        multilayer.layers[layer_index].n_spline_cubic = lambda x: np.maximum(
            original_n_spline_cubic(x), 0)
        original_k_spline_cubic = CubicSpline(wavelength, k_control)
        multilayer.layers[layer_index].k_spline_cubic = lambda x: np.maximum(
            original_k_spline_cubic(x), 0)
        multilayer.layers[layer_index].thickness = thickness

        # Calculate the residuals
        wavelength_range = data['wavelength']
        model_reflectance, _, _ = multilayer.calculate_RTA(wavelength_range)
        experimental_reflectance = data['reflectance']
        uncertainty = data['uncertainty']
        residuals = (1 / np.sqrt(len(wavelength_range) - len(params))) * ((1 / uncertainty) *
                                                                          ((model_reflectance - experimental_reflectance)))

        # Smooth regularization (minimize the difference between each two adjacent ns and ks)
        smooth_regularization = [
            weight_n * np.sum(np.diff(n_control)**2), weight_k * np.sum(np.diff(k_control)**2)]
        residuals = np.append(residuals, smooth_regularization)

        # Add second-order difference regularization for n and k
        second_diff_regularization = [
            weight_second_diff_n * np.sum(np.diff(n_control, n=2)**2),
            weight_second_diff_k * np.sum(np.diff(k_control, n=2)**2)
        ]
        residuals = np.append(residuals, second_diff_regularization)

        return residuals

    # Initialize the n, k, thickness
    initial_n = multilayer.layers[layer_index].initial_n
    initial_k = multilayer.layers[layer_index].initial_k
    initial_thickness = multilayer.layers[layer_index].thickness
    initial_params = [*initial_n, *initial_k, initial_thickness]

    print(
        f'Initial n of layer{layer_index} is: {multilayer.layers[layer_index].initial_n}')
    print(
        f'Initial k of layer{layer_index} is: {multilayer.layers[layer_index].initial_k}')

    # Set the lower and upper bounds for the parameters
    lower_bounds = [0]*n_points + [0]*n_points + [0]  # n, k, thickness
    upper_bounds = [3]*n_points + [3]*n_points + [1000]

    # Optimization using Levenberg-Marquardt algorithm (achieved by the least_squares(...) function)
    if smooth:
        data = data_smoothing(data)
        result = least_squares(residuals, initial_params,
                               bounds=(lower_bounds, upper_bounds), ftol=ftol)
    else:
        result = least_squares(residuals, initial_params,
                               bounds=(lower_bounds, upper_bounds), ftol=ftol)
    optimized_params = result.x

    # Setting optimized values
    multilayer.layers[layer_index].n = optimized_params[: n_points]
    multilayer.layers[layer_index].k = optimized_params[n_points: 2 * n_points]
    multilayer.layers[layer_index].thickness = optimized_params[-1]

    print('Optimal n: ', optimized_params[:n_points])
    print('Optimal k: ', optimized_params[n_points:2*n_points])
    print('Optimal Thickness: ', optimized_params[-1])

    wavelength_range = data['wavelength']
    R_optimized, _, _ = multilayer.calculate_RTA(wavelength_range)

    return R_optimized, optimized_params
