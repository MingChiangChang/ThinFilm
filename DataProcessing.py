import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import matplotlib.pyplot as plt


def process_data(path, left, right):
    """
    Read and crop the data within [left, right].

    **Parameters:**
    path : str
        path of the data to be processed.
    left : float
        The lower bound of the wavelength.
    right : float
        The upper bound of the wavelength.

    **Returns:**
    data: panda DataFrame
        It returns the cropped data.
    """
    # Load the data
    data = pd.read_csv(path, names=[
        '# lambda', 'reflectance', 'uncertainty', 'raw', 'dark', 'reference', 'fit'], skiprows=13)
    data = data.rename(columns={'# lambda': 'wavelength'})

    # Filter data based on wavelength
    data = data[(data['wavelength'] >= left) & (data['wavelength'] <= right)]

    return data


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
        'reflectance': smoothed_reflectance
    }
    smoothed_data = pd.DataFrame(smoothed_data)

    return smoothed_data


def plot_reflectance(data, smooth=False, multilayer=None, layer_index=None, optimal_data=None):
    """
    Plot selected datasets: original, smoothed, and/or optimal reflectance of a specific layer in a multilayer.

    Parameters:
    - data (pd.DataFrame): Experimental data where `data['wavelength']` is the wavelength, 
                           and `data['reflectance']` is the experimental reflectance.
    - multilayer (optional): The multilayer system object.
    - layer_index (int, optional): The index of the layer in the multilayer system for which the reflectance is being plotted.
    - smooth (bool, optional): Whether to smooth the data and plot the smoothed data or not. Defaults to False.
    - optimal_data (np.array, optional): The optimal reflectance data. Defaults to None.

    Returns:
    None. Displays the plot.
    """
    plt.figure()

    # Get the common wavelength range from the data
    wavelength_range = data['wavelength']

    # Plot the experimental reflectance
    plt.plot(wavelength_range, data['reflectance'],
             label='Original', alpha=0.3)

    # Plot smoothed data
    if smooth:
        plt.plot(wavelength_range, data_smoothing(data)[
                 'reflectance'], label='Smoothed', alpha=0.5)

    # Plot calculated reflectance
    if multilayer is not None and layer_index is not None:
        calculated_data, _, _ = multilayer.calculate_RTA(wavelength_range)
        material = multilayer.layers[layer_index].material
        thickness = multilayer.layers[layer_index].thickness
        plt.plot(wavelength_range, calculated_data,
                 label='Calculated', alpha=1.0)
        plt.title(
            f'Reflectance of multilayer with {material} at thickness = {thickness:.2f}nm')

    # Plot optimal reflectance
        if optimal_data is not None and optimal_data.size > 0:
            plt.plot(wavelength_range, optimal_data,
                     label='Optimal', alpha=1.0)
    else:
        plt.title('Experimental Reflectance')

    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.tight_layout()
    plt.show()
