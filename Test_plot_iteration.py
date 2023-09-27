from ThinFilmClasses import ThinFilmLayer, ThinFilmSystem
from DataProcessing import process_data, data_smoothing, plot_reflectance, plot_nk, extract_layer_thicknesses
from Optimization import optimize_nk
import time

# We want to save the plots of R and nk at every iteration.

# 1. Set the paths to save the plots
REFLECTANCE_PLOT_PATH = "R_nk_plot_gif/R_gif/"
NK_PLOT_PATH = "R_nk_plot_gif/nk_gif/"

# 2. Provide the path of the data, lower bound of the wavelength (left), and upper bound of the wavelength (right).
path1 = r"D:\OneDrive - Cornell University\Hongrui\Cornell\Master's project\Code\Reflectance Fitting\bi2o3\set2\film_data_0.2.csv"
data1, left, right = process_data(path1, uncertainty_threshold=0.005)
thickness1 = extract_layer_thicknesses(path1)
print(thickness1)

# 3. Create multilayer system
n_points1 = 10  # Number of points used to fit the n_spline and k_spline
air = ThinFilmLayer("air", 1, 0, left, right)
layer1 = ThinFilmLayer("Bi2O3", thickness1[0], n_points1, left, right)
layer2 = ThinFilmLayer("sio2", thickness1[1], n_points1, left, right)
substrate = ThinFilmLayer("c-Si", 1, 0, left, right)

multilayer1 = ThinFilmSystem([air, layer1, layer2, substrate])
initial_reflectance1, _, _ = multilayer1.calculate_RTA(data1['wavelength'])

# 4. Optimization, but now save the plots of R and nk at every iteration.
start_time = time.time()  # Record the start time before the optimization begins

# Plot the R and n,k before iteration begins (iteration 0)
# _________________________________________________________________________________________________________________
# Get the initial reflectance
initial_reflectance, _, _ = multilayer1.calculate_RTA(data1['wavelength'])

# Plot and save the initial reflectance
reflectance_filename = "reflectance_plot_0.png"
plot_reflectance(data1, multilayer1, layer_index=1, calculated_data=initial_reflectance, optimal_data=None,
                 iteration=0, elapsed_time=0, save=True, show=False, filepath=REFLECTANCE_PLOT_PATH, filename=reflectance_filename)

# Plot and save the initial n and k values
initial_params = [*multilayer1.layers[1].initial_n, *
                  multilayer1.layers[1].initial_k, multilayer1.layers[1].thickness]
nk_filename = "nk_plot_0.png"
plot_nk(data1, initial_params, n_points1, multilayer1, layer_index=1, iteration=0, elapsed_time=0,
        save=True, show=False, filepath=NK_PLOT_PATH, filename=nk_filename)
# __________________________________________________________________________________________________________________

# Do the optimization
optimal_data1, optimized_params1 = optimize_nk(multilayer1, layer_index=1, data=data1, n_points=n_points1, weight_n=1, weight_k=10e3, weight_second_diff_n=1,
                                               weight_second_diff_k=10e3, smooth=False, initial_reflectance=initial_reflectance1, reflectance_plot_path=REFLECTANCE_PLOT_PATH, nk_plot_path=NK_PLOT_PATH, start_time=start_time)
