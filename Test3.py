from ThinFilmClasses import ThinFilmLayer, ThinFilmSystem
from DataProcessing import process_data, data_smoothing, plot_reflectance, plot_nk, extract_layer_thicknesses
from Optimization import optimize_nk
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Directory path
dir_path = r"D:\OneDrive - Cornell University\Hongrui\Cornell\Master's project\Code\Reflectance Fitting\bi2o3\set2"

# Lists to store results
all_optimal_data = []
all_n = []
all_k = []
all_n_spline = []
all_k_spline = []
failed_files = []

# Process all CSV files in the directory
for filename in sorted(os.listdir(dir_path)):
    if filename.endswith(".csv"):
        path = os.path.join(dir_path, filename)
        print(f"Processing file: {filename}")
        try:
            data, left, right = process_data(
                path, left=400, right=700, uncertainty_threshold=None)
            thickness = extract_layer_thicknesses(path)
            wavelength_range = data['wavelength']

            # Create multilayer system
            n_points = 10  # Number of points used to fit the n_spline and k_spline
            air = ThinFilmLayer("air", 1, 0, left, right)
            layer1 = ThinFilmLayer(
                "Bi2O3", thickness[0], n_points, left, right)
            layer2 = ThinFilmLayer("sio2", thickness[1], n_points, left, right)
            substrate = ThinFilmLayer("c-Si", 1, 0, left, right)

            multilayer = ThinFilmSystem([air, layer1, layer2, substrate])

            # Optimization
            optimal_data, optimized_params = optimize_nk(multilayer, layer_index=1, data=data, n_points=n_points,
                                                         weight_n=1, weight_k=10e3, weight_second_diff_n=1, weight_second_diff_k=10e3, ftol=10e-4)

            # Append the results to the lists
            all_optimal_data.append(optimal_data)
            all_n.append(optimized_params[:n_points])
            all_k.append(optimized_params[n_points: 2 * n_points])
            n_spline_original = CubicSpline(
                layer1.wavelength, optimized_params[:n_points])
            k_spline_original = CubicSpline(
                layer1.wavelength, optimized_params[n_points: 2 * n_points])

            # CubicSpline n, k. Alternatively, can use the lambda expression
            def n_spline_cubic(x): return np.maximum(n_spline_original(x), 0)
            def k_spline_cubic(x): return np.maximum(k_spline_original(x), 0)
            # use this one to plot heatmap
            all_n_spline.append(n_spline_cubic(wavelength_range))
            # use this one to plot heatmap
            all_k_spline.append(k_spline_cubic(wavelength_range))
            print()
            # print(all_optimal_data)
            # print(all_n)
            # print(all_k)
        except Exception as e:
            print(f"Failed to process {filename} due to {str(e)}")
            failed_files.append(filename)


def plot_heatmap(data, title, xlabel, ylabel, zlabel, x_axis_values=None, y_axis_values=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    cbar = fig.colorbar(cax, label=zlabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjusting x-ticks for better alignment with data columns
    if x_axis_values is not None:
        ax.set_xticks(np.arange(len(x_axis_values)))
        ax.set_xticklabels(x_axis_values, rotation=45, ha="left")

    if y_axis_values is not None:
        num_ticks = 10  # or any other number that you think is appropriate
        indices = np.linspace(0, len(y_axis_values)-1,
                              num_ticks, dtype=int)  # indices for ticks
        labels = [str(y_axis_values[i])
                  for i in indices]  # actual wavelength values for labels
        ax.set_yticks(indices)
        ax.set_yticklabels(labels)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()  # Close the figure if saving, so it doesn't display again
    else:
        plt.show()


def plot_all_reflectance_2D(all_data, save_path=None):
    plt.figure(figsize=(8, 6))
    for idx, reflectance_data in enumerate(all_data):
        plt.plot(data['wavelength'], reflectance_data, label=f"File {idx+1}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('All Reflectance Datasets')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_all_n_2D(all_n_data, save_path=None):
    plt.figure(figsize=(8, 6))
    for idx, n_data in enumerate(all_n_data):
        plt.plot(data['wavelength'], n_data, label=f"File {idx+1}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Refractive Index (n)')
    plt.title('All n Datasets')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_all_k_2D(all_k_data, save_path=None):
    plt.figure(figsize=(8, 6))
    for idx, k_data in enumerate(all_k_data):
        plt.plot(data['wavelength'], k_data, label=f"File {idx+1}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Extinction Coefficient (k)')
    plt.title('All k Datasets')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        plt.show()


# Assuming the y-axis is a common wavelength range from your data
# wavelength_range = data['wavelength']
x_positions = range(len(all_optimal_data))

# Define paths for saving the heatmaps
# Modify this to your desired directory
output_dir = r"D:\OneDrive - Cornell University\Hongrui\Cornell\Master's project\Code\Reflectance Fitting\Plots\heatmap"
optimal_data_save_path = os.path.join(
    output_dir, 'optimal_reflectance_heatmap.png')
n_save_path = os.path.join(output_dir, 'refractive_index_heatmap.png')
k_save_path = os.path.join(output_dir, 'extinction_coefficient_heatmap.png')

# Plot and save heatmaps
# Transpose since imshow displays the axis in reverse order
all_optimal_data_array = np.vstack(all_optimal_data).T
all_n_array = np.vstack(all_n_spline).T
all_k_array = np.vstack(all_k_spline).T
plot_heatmap(all_optimal_data_array, 'Optimal Reflectance', 'X Position',
             'Wavelength [nm]', 'Reflectance', save_path=optimal_data_save_path)
plot_heatmap(all_n_array, 'Refractive Index (n)',
             'X Position', 'Wavelength [nm]', 'n', save_path=n_save_path)
plot_heatmap(all_k_array, 'Extinction Coefficient (k)',
             'X Position', 'Wavelength [nm]', 'k', save_path=k_save_path)


def plot_all_reflectance_2D(all_data, save_path=None):
    plt.figure(figsize=(8, 6))
    for idx, reflectance_data in enumerate(all_data):
        plt.plot(data['wavelength'], reflectance_data, label=f"File {idx+1}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('All Reflectance Datasets')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_all_n_2D(all_n_data, save_path=None):
    plt.figure(figsize=(8, 6))
    for idx, n_data in enumerate(all_n_data):
        plt.plot(data['wavelength'], n_data, label=f"File {idx+1}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Refractive Index (n)')
    plt.title('All n Datasets')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_all_k_2D(all_k_data, save_path=None):
    plt.figure(figsize=(8, 6))
    for idx, k_data in enumerate(all_k_data):
        plt.plot(data['wavelength'], k_data, label=f"File {idx+1}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Extinction Coefficient (k)')
    plt.title('All k Datasets')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        plt.show()


# 2d plots
# For Reflectance
# output_dir = r"D:\OneDrive - Cornell University\Hongrui\Cornell\Master's project\Code\Reflectance Fitting\Plots\2Dplot"
# save_path = os.path.join(output_dir, 'all_reflectance_plot.png')
# plot_all_reflectance_2D(all_optimal_data, save_path)

# # For n
# save_path = os.path.join(output_dir, 'all_n_plot.png')
# plot_all_n_2D(all_n_spline, save_path)

# # For k
# save_path = os.path.join(output_dir, 'all_k_plot.png')
# plot_all_k_2D(all_k_spline, save_path)

# # Display names of failed files
# if failed_files:
#     print("Failed to process the following files:")
#     for file in failed_files:
#         print(file)
