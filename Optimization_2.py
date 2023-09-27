from scipy.optimize import least_squares
from DataProcessing import data_smoothing
from scipy.interpolate import CubicSpline
import numpy as np
import torch
from torch.autograd import Variable

# Define the residual function


def residuals(params, multilayer, layer_index, data, n_points, weight_n=1, weight_k=1):
    """
    Calculate the residuals between calculated reflectance and experimental reflectance.

    Parameters:
    - params (list of float): List of parameters [n1, n2, ..., k1, k2, ..., thickness] to be optimized.
    Returns:
    - list of float: Residuals = calculated_reflectance - experimental_reflectance.
    """
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
                                                                      (model_reflectance - experimental_reflectance))

    # Smooth regularization (minimize the difference between each two adjacent ns and ks)
    smooth_regularization = [
        weight_n * np.sum(np.diff(n_control)**2), weight_k * np.sum(np.diff(k_control)**2)]
    residuals = np.append(residuals, smooth_regularization)

    return residuals


# Levenberg-Marquardt algorithm
def optimize_nk_LM(residuals, multilayer, layer_index, data, n_points, weight_n=1, weight_k=1, smooth=False):
    """
    Optimize the control points of n, k, and the thickness for a specific layer in the multilayer using Levenberg-Marquardt algorithm.

    Parameters:
    - multilayer (ThinFilmSystem): The multilayer system object.
    - layer_index (int): The index of the layer in the multilayer system to be optimized.
    - data (pd.DataFrame): Experimental data where `data['wavelength']` is the wavelength and `data['reflectance']` is the experimental reflectance.
    - n_points (int): Number of points used to fit n, k.
    - smooth_reg_factor (float or int, optional): A factor to minimize the difference between two adjacent ns and ks so that the n, k function are smoother. Default is 100.
    - weight_n: Weight of the sum of the difference between two adjacent ns. Higher the weight, emphasize more on the smoothing of n.
    - weight_k: Weight of the sum of the difference between two adjacent ks. Higher the weight, emphasize more on the smoothing of k.

    Returns:
    - optimized_params (list of float): A list containing the optimized values [n1, n2, ..., k1, k2, ..., thickness].
    - R_optimized (list of float): The reflectance calculated using the optimized parameters.
    """

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
        result = least_squares(residuals, initial_params, args=(multilayer, layer_index, data, n_points, weight_n, weight_k),
                               bounds=(lower_bounds, upper_bounds))
    else:
        result = least_squares(residuals, initial_params, args=(multilayer, layer_index, data, n_points, weight_n, weight_k),
                               bounds=(lower_bounds, upper_bounds))
    print(result.nfev)
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


# Adaptive Moment Estimation (Adam) algorithm
