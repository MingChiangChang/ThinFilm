# This file processes the reflectance data (crop, smooth, extract layer thickness), and plot the reflectance and n, k.

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import os


def process_data(path, left, right, uncertainty_threshold=0.02):
    """
    Read and crop the data within [left, right], and return cropped data, left, right.

    **Parameters:**
    path : str
        path of the data to be processed.
    uncertainty_threshold : float
        uncertainty threshold below which the data is selected.
    left: float
        Manually set the lower bound of the cropped data.
    right: float
        Manually set the upper bound of the cropped data.
    **Returns:**
    data : panda DataFrame
        It returns the cropped data.
    left : float
        The lower bound of the cropped data based on uncertainty threshold.
    right : float
        The upper bound of the cropped data based on uncertainty threshold.
    """
    # Load the data
    data = pd.read_csv(path, names=[
        '# lambda', 'reflectance', 'uncertainty', 'raw', 'dark', 'reference', 'fit'], skiprows=13)
    data = data.rename(columns={'# lambda': 'wavelength'})

    # Filter data based on uncertainty
    if uncertainty_threshold != None and uncertainty_threshold > 0:
        uncertainty_filtered_data = data[data['uncertainty']
                                         <= uncertainty_threshold]

        # Determine the smallest and largest wavelengths
        left = uncertainty_filtered_data['wavelength'].min()
        right = uncertainty_filtered_data['wavelength'].max()

        print("left = ", left)
        print("right = ", right)

        # Filter data based on determined wavelength boundaries
        data = data[(data['wavelength'] >= left) &
                    (data['wavelength'] <= right)]
    else:
        data = data[(data['wavelength'] >= left) &
                    (data['wavelength'] <= right)]

    return data, left, right


def extract_layer_thicknesses(file_path):
    """
    Extract the layer thicknesses of materials from the CSV file.

    The function reads a file and extracts information between
    "# SAMPLE STACK" and the subsequent "#   substrate ...". 
    For all rows within this section (except for the substrate),
    the function captures the thickness value of each material.

    **Parameters:**
    - file_path (str): The path to the CSV file.

    **Returns:**
    - list of float: A list containing the thickness values of each material.
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    sample_data_lines = []
    capture = False  # Flag to determine whether we should capture lines

    for line in lines:
        stripped_line = line.strip().replace(",", "")

        if "# SAMPLE STACK" in stripped_line:
            capture = True  # Start capturing after encountering "# SAMPLE STACK"
            continue
        elif capture and "#   substrate" in stripped_line:
            break
        elif "# END" in stripped_line:  # Handle cases where "# END" comes before the "# SAMPLE STACK"
            capture = False  # Reset the flag
            continue
        elif capture:
            sample_data_lines.append(stripped_line)

    # Print the lines between "# SAMPLE STACK" and "#   substrate ..."
    print("\nCaptured lines:")
    for line in sample_data_lines:
        print(line.strip())

    layer_thicknesses = []

    for line in sample_data_lines:
        grids = line.split()
        thickness = float(grids[3])
        layer_thicknesses.append(thickness)

    return layer_thicknesses


def data_smoothing(data, kernel_size=11, sigma=20, gaussian_filter=True):
    """
    Apply 1.median filter 2.gaussian_filter1d to the data.

    **Parameters:**
    data : panda DataFrame
        Processed data to be smoothed.
    kernel_size : int
        kernel size of the median filter. It needs to be an odd integer. 
        The median filter is used to remove outliers.
        Larger the kernel size, better the smoothing.
    sigma : int
        Standard Deviation for the gaussian kernel.
        The 1d gaussian filter smoothes the data.
        Larger the sigma, better the smoothing.
    gaussian_filter : bool
        If true, then gaussian_filter_1d is applied.

    **Returns:**
    smoothed_data: panda DataFrame
        It returns the smoothed data.
    """
    # Apply the moving median filter to remove sudden spikes
    data_without_outliers = data['reflectance']
    data_without_outliers = medfilt(
        data['reflectance'], kernel_size=kernel_size)

    if gaussian_filter == True:
        # Apply a Gaussian smoothing
        smoothed_reflectance = gaussian_filter1d(data_without_outliers, sigma)

    # Create a smoothed panda DataFrame
    smoothed_data = {
        'wavelength': data['wavelength'],
        'reflectance': smoothed_reflectance,
        'uncertainty': data['uncertainty']
    }
    smoothed_data = pd.DataFrame(smoothed_data)

    return smoothed_data


def plot_reflectance(data, multilayer, layer_index=1, smooth=False, calculated_data=None, optimal_data=None, tfoc_data=False, save=False, show=True, filepath='', filename='reflectance_plot.png', iteration=None, elapsed_time=None):
    """
    Plot selected datasets: original, smoothed, and/or optimal reflectance of a specific layer in a multilayer.

    Parameters:
    - data (pd.DataFrame): Experimental data with columns 'wavelength' and 'reflectance' representing the wavelength and experimental reflectance respectively.
    - multilayer (object): A multilayer object containing layer information.
    - layer_index (int, optional): Index of the specific layer in the multilayer to be plotted. Defaults to 1.
    - smooth (bool, optional): Whether to smooth the data and plot the smoothed data. Defaults to False.
    - calculated_data (np.array, optional): The calculated reflectance data (before optimization). Defaults to None.
    - optimal_data (np.array, optional): The optimal reflectance data. Defaults to None.
    - save (bool, optional): Whether to save the generated plot to a file. Defaults to False.
    - filepath (str, optional): Directory path where the plot will be saved. Defaults to the current working directory.
    - filename (str, optional): Name of the file to save the plot as. Defaults to 'reflectance_plot.png'.

    Returns:
    - None. Displays or saves the plot based on provided parameters.
    """
    plt.figure(figsize=(8, 6))

    # Get the common wavelength range from the data
    wavelength_range = data['wavelength']

    # Update the title based on what is plotted
    title = 'Experimental Reflectance'

    # Plot the experimental reflectance
    plt.plot(wavelength_range, data['reflectance'],
             label='Original', alpha=0.3, color='#1f77b4')

    # Get the material name and thickness from multilayer
    material = multilayer.layers[layer_index].material
    thickness = multilayer.layers[layer_index].thickness

    # Plot smoothed data
    if smooth:
        plt.plot(wavelength_range, data_smoothing(data)[
                 'reflectance'], label='Smoothed', alpha=0.3, color='#2ca02c')

    # Plot calculated reflectance
    if calculated_data is not None and calculated_data.size > 0:
        plt.plot(wavelength_range, calculated_data,
                 label='Calculated', alpha=1.0, color='#ff7f0e')
        title = f'Reflectance of multilayer with {material} at thickness = {thickness:.2f}nm'

    # Plot optimal reflectance
    if optimal_data is not None and optimal_data.size > 0:
        plt.plot(wavelength_range, optimal_data,
                 label='Optimal', alpha=1.0, color='#d62728')
        title = f'Reflectance of multilayer with {material} at thickness = {thickness:.2f}nm'
        if iteration is not None and elapsed_time is not None:
            title = f'Reflectance of multilayer with {material} at thickness = {thickness:.2f}nm\n - Iteration {iteration}\n - Elapsed Time: {elapsed_time:.2f}s'

    # Plot tfoc data (data['fit'])
    if tfoc_data:
        plt.plot(wavelength_range, data['fit'], label='tfoc')

    plt.title(title, fontsize=10)
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.tight_layout()

    # Choose to save the plot or not
    plt.tight_layout()
    if save:
        complete_path = os.path.join(filepath, filename)
        plt.savefig(complete_path, dpi=300)

    # Choose to show the plot or not
    if show:
        plt.show()
    else:
        plt.close()


def plot_nk(data, optimized_params, n_points, multilayer, layer_index=1, save=False, show=True, filepath='', filename='nk_plot.png', iteration=None, elapsed_time=None):
    """
    Plot refractive index (n) and extinction coefficient (k) before and after optimization for a specific material layer.

    Parameters:
    - data (pd.DataFrame): Experimental data with a 'wavelength' column representing the wavelength range.
    - optimized_params (np.array): Array containing optimized values of refractive index and extinction coefficient.
    - n_points (int): Number of data points in the 'n' or 'k' dataset.
    - multilayer (object): A multilayer object containing layer information.
    - layer_index (int, optional): Index of the specific layer in the multilayer whose properties are to be plotted. Defaults to 1.
    - save (bool, optional): Whether to save the generated plot to a file. Defaults to False.
    - filepath (str, optional): Directory path where the plot will be saved. Defaults to the current working directory.
    - filename (str, optional): Name of the file to save the plot as. Defaults to 'nk_plot.png'.

    Returns:
    - None. Displays or saves the plot based on provided parameters.
    """
    # Get the wavelength range from the data
    wavelength_range = data['wavelength']

    # Get the material name from multilayer
    layer = multilayer.layers[layer_index]
    material = layer.material

    # Extract the n, k before and after optimization
    n_initial = layer.initial_n
    k_initial = layer.initial_k
    n_optimal = optimized_params[: n_points]
    k_optimal = optimized_params[n_points: 2 * n_points]

    # Create a figure and a 2x1 grid of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))

    # 1. Plot n on the first subplot (ax1)

    # Scatter plot of initial n
    ax1.scatter(layer.wavelength, n_initial, c='r')
    # CubicSpline plot of initial n
    n_spline_cubic_original_obj = CubicSpline(layer.wavelength, n_initial)
    n_spline_data = np.maximum(
        n_spline_cubic_original_obj(wavelength_range), 0)
    ax1.plot(wavelength_range, n_spline_data, c='r', label='initial n')

    # Scatter plot of optimal n
    ax1.scatter(layer.wavelength, n_optimal, c='b')
    # CubicSpline plot of optimal n
    n_spline_cubic_optimial_obj = CubicSpline(layer.wavelength, n_optimal)
    n_spline_data_optimal = np.maximum(
        n_spline_cubic_optimial_obj(wavelength_range), 0)
    ax1.plot(wavelength_range, n_spline_data_optimal,
             c='b', label='optimial n')

    # Set label, legend, and title for ax1
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Refractive index (n)')
    ax1.legend()
    ax1.set_title(
        f'n vs wavelength for {material}', fontsize=10)

    # 2. Plot k on the second subplot (ax2)

    # Scatter plot of initial k
    ax2.scatter(layer.wavelength, k_initial, c='r')
    # CubicSpline plot of initial k
    k_spline_cubic_original_obj = CubicSpline(layer.wavelength, k_initial)
    k_spline_data = np.maximum(
        k_spline_cubic_original_obj(wavelength_range), 0)
    ax2.plot(wavelength_range, k_spline_data, c='r', label='initial k')

    # Scatter plot of optimal k
    ax2.scatter(layer.wavelength, k_optimal, c='b')
    # CubicSpline plot of optimal k
    k_spline_cubic_optimized_obj = CubicSpline(layer.wavelength, k_optimal)
    k_spline_data_optimized = np.maximum(
        k_spline_cubic_optimized_obj(wavelength_range), 0)
    ax2.plot(wavelength_range, k_spline_data_optimized,
             c='b', label='optimial k')

    # Set label, legend, and title for ax2
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Extinction Coefficient (k)')
    ax2.legend()
    ax2.set_title(
        f'k vs wavelength for {material}', fontsize=10)

    if iteration is not None and elapsed_time is not None:
        ax1.set_title(
            f'n vs wavelength for {material} \n- Iteration {iteration} \n- Elapsed Time: {elapsed_time:.2f}s', fontsize=10)
        ax2.set_title(
            f'k vs wavelength for {material} \n- Iteration {iteration} \n- Elapsed Time: {elapsed_time:.2f}s', fontsize=10)

    plt.tight_layout()

    # Choose to save the plot or not
    if save:
        complete_path = os.path.join(filepath, filename)
        plt.savefig(complete_path, dpi=300)

    # Choose to show the plot or not
    if show:
        plt.show()
    else:
        plt.close()
