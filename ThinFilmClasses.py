import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from copy import deepcopy
import sys


class ThinFilmLayer:
    """
    Class to represent a single layer in a thin film system.
    It encapsulates the layer's index of refraction 'n', extinction coefficient 'k', 
    and thickness. n and k vary with wavelength.
    """

    @staticmethod
    def find_file_insensitive(path, filename):
        """
        Search for a file in the specified path in a case-insensitive manner.

        Parameters:
        path : str
            The directory path where to look for the file.
        filename : str
            The name of the file to look for.

        Returns:
        str or None
            The name of the file with the original case if found, None otherwise.
        """
        filename_lower = filename.lower()
        for file in os.listdir(path):
            if file.lower() == filename_lower:
                return file
        return None

    def __init__(self, material, thickness, n_points, min_wavelength, max_wavelength):
        """
        Initialize the layer with the chosen material's properties.

        Parameters:
        material : str
            Name of the material where its n, k are extracted.
        thickness : float
            Thickness of the material chosen [nm]. 
            For air and substrate, the thickness does not matter (can be set to 1, for example).
        n_points: int
            Number of points for spline fitting.
            If n_points == 0, then use all the n and k data within the wavelength range.
        min_wavelength: float
            Lower bound of the wavelength used to fit for n and k.
        max_wavelength: float
            Upper bound of the wavelength used to fit for n and k.
        """
        # Assertions to check input types
        assert isinstance(material, str), "material should be a string"
        assert isinstance(thickness, (float, int)
                          ), "thickness should be a float or int"
        assert isinstance(n_points, int), "n_points should be an integer"
        assert isinstance(min_wavelength, (float, int)
                          ), "min_wavelength should be a float or int"
        assert isinstance(max_wavelength, (float, int)
                          ), "max_wavelength should be a float or int"

        # Additional assertions for value constraints
        assert thickness >= 0, "thickness should be non-negative"
        assert n_points >= 0, "n_points should be non-negative"
        assert min_wavelength >= 0 and min_wavelength < max_wavelength, "min_wavelength should be non-negative and less than max_wavelength"
        assert max_wavelength >= 0 and max_wavelength > min_wavelength, "max_wavelength should be non-negative and greater than min_wavelength"

        self.material = material
        self.thickness = thickness

        # check if a file with material name exists
        path = "database_nk"
        filename = self.find_file_insensitive(path, f"{material}.csv")
        if filename is not None:
            # if yes, read the file
            data = pd.read_csv(os.path.join(path, filename))
            print(f"nk data found for {material}.")

            photon_energy = data.iloc[:, 0].values
            wavelength = 1239.8419843320021 / photon_energy  # an array of wavelengths
            n = data.iloc[:, 1].values
            k = data.iloc[:, 2].values

            # Sort wavelength, n, and k in ascending order for spline fitting
            sorted_indices = np.argsort(wavelength)
            wavelength = wavelength[sorted_indices]
            n = n[sorted_indices]
            k = k[sorted_indices]

        else:
            # if the file does not exist, generate n and k
            wavelength = np.linspace(min_wavelength, max_wavelength, n_points)
            n = np.random.uniform(1.4, 2.0, n_points)
            k = np.random.uniform(0.0, 0.2, n_points)
            print(f"nk data not found for {material}.", file=sys.stderr)

        # Filtering the data based on wavelength range before creating the spline representation
        indices = (wavelength >= min_wavelength) & (
            wavelength <= max_wavelength)
        wavelength = wavelength[indices]
        n = n[indices]
        k = k[indices]

        # If n_points != 0, select n points with approximately equal spacing in the wavelength range
        # (If n_points == 0, use all the data points)
        if n_points != 0:
            indices = np.round(np.linspace(
                0, len(wavelength) - 1, n_points)).astype(int)
            wavelength = wavelength[indices]
            n = n[indices]
            k = k[indices]

        self.wavelength = wavelength
        self.n = np.maximum(n, 0)
        self.k = np.maximum(k, 0)

        # Store deep copies of the initial values
        self.initial_n = deepcopy(self.n)
        self.initial_k = deepcopy(self.k)

        # Generate spline representation of n and k
        original_n_spline_cubic = CubicSpline(wavelength, self.n)
        self.n_spline_cubic = lambda x: np.maximum(
            original_n_spline_cubic(x), 0)

        original_k_spline_cubic = CubicSpline(wavelength, self.k)
        self.k_spline_cubic = lambda x: np.maximum(
            original_k_spline_cubic(x), 0)

    def get_n(self, wavelength):
        """
        Returns the index of refraction at the specified wavelength using spline interpolation.

        Parameter:
        wavelength : float or array of floats
            The wavelength [nm] at which to get the index of refraction.
        """
        n_value = self.n_spline_cubic(wavelength)
        return np.maximum(n_value, 0)

    def get_k(self, wavelength):
        """
        Returns the extinction coefficient at the specified wavelength using spline interpolation.

        Parameter:
        wavelength : float or array of floats
            The wavelength [nm] at which to get the extinction coefficient.
        """
        k_value = self.k_spline_cubic(wavelength)
        return np.maximum(k_value, 0)

    def get_N(self, wavelength):
        """
        Returns the refractive index at the specified wavelength using spline interpolation.

        Parameter:
        wavelength : float or array of floats
            The wavelength [nm] at which to get the refractive index.
        """
        return self.get_n(wavelength) + 1j * self.get_k(wavelength)

    def set_n(self, wavelength, new_n_value):
        """
        Sets the refractive index at the specified wavelength and updates the spline representation.

        Parameters:
        wavelength : float
            The wavelength [nm] at which to set the index of refraction.
            It is not necessarily the exact wavelength, just need to be the wavelength that is near the point to be adjusted.
        new_n_value : float
            The new index of refraction value.
        """
        idx = (np.abs(self.wavelength - wavelength)
               ).argmin()  # find the closest wavelength
        self.n[idx] = new_n_value
        original_n_spline_cubic = CubicSpline(self.wavelength, self.n)
        self.n_spline_cubic = lambda x: np.maximum(
            original_n_spline_cubic(x), 0)

    def set_k(self, wavelength, new_k_value):
        """
        Sets the extinction coefficient at the specified wavelength and updates the spline representation.

        Parameters:
        wavelength : float
            The wavelength [nm] at which to set the extinction coefficient.
            It is not necessarily the exact wavelength, just need to be the wavelength that is near the point to be adjusted.
        new_k_value : float
            The new extinction coefficient value.
        """
        idx = (np.abs(self.wavelength - wavelength)
               ).argmin()  # find the closest wavelength
        self.k[idx] = new_k_value
        original_k_spline_cubic = CubicSpline(self.wavelength, self.k)
        self.k_spline_cubic = lambda x: np.maximum(
            original_k_spline_cubic(x), 0)

    def reset_n_k(self):
        """
        Reset the refractive index (n) and extinction coefficient (k) 
        to their original values when the instance was first created.
        """
        self.n = deepcopy(self.initial_n)
        self.k = deepcopy(self.initial_k)


# **Plotting methods**

    def plot_n(self, min_wavelength, max_wavelength):
        """
        Plots the index of refraction as a function of wavelength.

        Parameters:
        min_wavelength: float
            Lower bound of the wavelength used to plot n.
        max_wavelength: float
            Upper bound of the wavelength used to plot n.
        """
        wavelengths = np.linspace(min_wavelength, max_wavelength, num=1000)
        plt.figure()
        plt.plot(wavelengths, self.n_spline_cubic(
            wavelengths), '-', label='n spline cubic')
        plt.scatter(self.wavelength, self.n, marker='o',
                    color='r', label='n data')
        plt.xlabel('Wavelength (λ) [nm]')
        plt.ylabel('Refractive index (n)')
        plt.title(f'Refractive index (n) vs wavelength for {self.material}')
        plt.legend()
        plt.show()

    def plot_k(self, min_wavelength, max_wavelength):
        """
        Plots the extinction coefficient as a function of wavelength.

        Parameters:
        min_wavelength: float
            Lower bound of the wavelength used to plot n.
        max_wavelength: float
            Upper bound of the wavelength used to plot n.
        """
        wavelengths = np.linspace(min_wavelength, max_wavelength, num=1000)
        plt.figure()
        plt.plot(wavelengths, self.k_spline_cubic(
            wavelengths), '-', label='k spline cubic')
        plt.scatter(self.wavelength, self.k, marker='o',
                    color='r', label='k data')
        plt.xlabel('Wavelength (λ) [nm]')
        plt.ylabel('Extinction coefficient (k)')
        plt.title(
            f'Extinction coefficient (k) vs wavelength for {self.material}')
        plt.legend()
        plt.show()

    def plot_nk(self, min_wavelength, max_wavelength):
        """
        Plots the index of refraction and extinction coefficient as functions of wavelength,
        both for the input data and the spline interpolation.

        Parameters:
        min_wavelength: float
            Lower bound of the wavelength used to plot n and k.
        max_wavelength: float
            Upper bound of the wavelength used to plot n and k.
        """
        wavelengths = np.linspace(min_wavelength, max_wavelength, num=1000)
        n_values = self.n_spline_cubic(wavelengths)
        k_values = self.k_spline_cubic(wavelengths)

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Wavelength (λ) [nm]')
        ax1.set_ylabel('Refractive index (n)', color='tab:blue')
        ax1.plot(wavelengths, n_values, '-',
                 color='tab:blue', label='n spline')
        ax1.scatter(self.wavelength, self.n, marker='o',
                    color='r', label='n data')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Extinction coefficient (k)', color='tab:red')
        ax2.plot(wavelengths, k_values, '-', color='tab:red', label='k spline')
        ax2.scatter(self.wavelength, self.k, marker='o',
                    color='r', label='k data')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title(
            f'Refractive index and extinction coefficient vs wavelength for {self.material}')
        fig.legend(loc="upper right", bbox_to_anchor=(
            1, 1), bbox_transform=ax1.transAxes)
        plt.show()


class ThinFilmSystem:
    """
    Class to represent a thin film system composed of multiple layers.
    """

    def __init__(self, layers):
        """
        Initialize the system with a list of ThinFilmLayer objects.
        layers : list of ThinFilmLayer
            Layers of the system in order from the top layer to the substrate.
        """
        assert isinstance(layers, list) and all(isinstance(layer, ThinFilmLayer)
                                                for layer in layers), "layers should be a list of ThinFilmLayer objects"
        self.layers = layers

    def add_layer(self, layer):
        """
        Add a layer to the system. The new layer is added to the top of the stack.

        Parameters:
        layer : ThinFilmLayer
            The layer to add. This should be an instance of the ThinFilmLayer class.
        """
        self.layers.insert(0, layer)

# **Calculation methods**

    def calculate_propagation_matrix(self, wavelength, layer_index):
        """
        Calculate the propagation matrix for this layer for a particular wavelength.
        Assume the incident light is normal (or the calculation would be very complicated...)
        Can add the non-normal part later using Snell's law: N1cos(θ1) = N2cos(θ2)

        Parameters:
        wavelength : float
            Wavelength [nm] at which to calculate the propagation matrix of this layer.
        layer_index: int
            Index of the layer to be calculated in the multilayer system.

        Returns:
        P : 2D array
            Propagation matrix of this layer.
        """
        # If wavelength is a float or int, execute original code
        if isinstance(wavelength, (float, int)):
            N = self.layers[layer_index].get_N(wavelength)
            thickness = self.layers[layer_index].thickness
            phase = 2 * np.pi * N * thickness / wavelength
            P = np.array([[np.exp(1j * phase.real) * np.exp(-phase.imag), 0],
                          [0, np.exp(-1j * phase.real) * np.exp(phase.imag)]])
        # If wavelength is an array, execute vectorized code
        elif isinstance(wavelength, np.ndarray):
            N = np.array([self.layers[layer_index].get_N(w)
                         for w in wavelength])
            thickness = self.layers[layer_index].thickness
            phase = 2 * np.pi * N * \
                thickness[:, np.newaxis] / wavelength[:, np.newaxis]
            P = np.array([[np.exp(1j * phase.real) * np.exp(-phase.imag), np.zeros_like(phase)],
                          [np.zeros_like(phase), np.exp(-1j * phase.real) * np.exp(phase.imag)]])
            P = np.transpose(P, (2, 0, 1))
        else:
            raise ValueError("Invalid type for wavelength.")

        return P

    def calculate_boundary_matrix(self, wavelength, layer_index):
        """
        Calculate the boundary matrix for two adjacent layers ("this" layer and "next" layer) 
        in the multilayer system for a particular wavelength.
        Assume the incident light is normal (or the calculation would be very complicated...)
        Can add the non-normal part later using Snell's law: N1cos(θ1) = N2cos(θ2)

        Parameters:
        wavelength : float
            Wavelength [nm] at which to calculate the boundary matrix of this layer.
        layer_index: int
            Index of the layer to be calculated in the multilayer system.

        Returns:
        B : 2D array
            Boundary matrix of the boundary between this layer and the next layer.
        """
        # If wavelength is a float or int, execute original code
        if isinstance(wavelength, (float, int)):
            N_this = self.layers[layer_index].get_N(wavelength)
            N_next = self.layers[layer_index + 1].get_N(wavelength)
            B = 1 / (2 * N_this) * np.array([[N_this + N_next, N_this - N_next],
                                            [N_this - N_next, N_this + N_next]])
            B = np.conjugate(B)
        # If wavelength is an array, execute vectorized code
        elif isinstance(wavelength, np.ndarray):
            N_this = np.array([self.layers[layer_index].get_N(w)
                              for w in wavelength])
            N_next = np.array([self.layers[layer_index + 1].get_N(w)
                              for w in wavelength])
            B = 1 / (2 * N_this[:, np.newaxis]) * np.array([[N_this + N_next, N_this - N_next],
                                                            [N_this - N_next, N_this + N_next]])
            B = np.conjugate(B)
            B = np.transpose(B, (2, 0, 1))
        else:
            raise ValueError("Invalid type for wavelength.")

        return B

    def calculate_total_transfer_matrix(self, wavelength):
        """
        Calculate the total transfer matrix for the multilayer system 
        for a particular wavelength.
        Assume the incident light is normal (or the calculation would be very complicated...)
        Can add the non-normal part later using Snell's law: N1cos(θ1) = N2cos(θ2)

        Parameters:
        wavelength : float
            Wavelength [nm] at which to calculate the transfer matrix of this layer.

        Returns:
        M_total : 2D array
            Total transfer matrix of the multilayer system.
        """
        # Handle scalar wavelength
        if isinstance(wavelength, (float, int)):
            M = np.eye(2)
            B01 = self.calculate_boundary_matrix(wavelength, 0)
            M = M @ B01
            for index in range(1, len(self.layers) - 1):
                P = self.calculate_propagation_matrix(wavelength, index)
                B = self.calculate_boundary_matrix(wavelength, index)
                M = M @ P @ B
            return M

        # Handle array wavelength
        elif isinstance(wavelength, np.ndarray):
            num_wavelengths = len(wavelength)
            M = np.eye(2)[np.newaxis].repeat(num_wavelengths, axis=0)

            B01 = self.calculate_boundary_matrix(wavelength, 0)
            M = np.einsum('ijk,ikl->ijl', M, B01)

            for index in range(1, len(self.layers) - 1):
                P = self.calculate_propagation_matrix(wavelength, index)
                B = self.calculate_boundary_matrix(wavelength, index)
                M = np.einsum('ijk,ikl->ijl',
                              np.einsum('ijk,ikl->ijl', M, P), B)
            return M
        else:
            raise ValueError("Invalid type for wavelength.")

    def calculate_RTA(self, wavelength_range):
        """
        Calculate the reflectance (R), transmittance (T), and absorption (A)
        for a multilayer system as a function of wavelength.
        Assume the incident light is normal (or the calculation would be very complicated...)
        Can add the non-normal part later using Snell's law: N1cos(θ1) = N2cos(θ2)

        Parameters:
        wavelength_range : array-like
            Range of wavelengths [nm] for which R, T, A should be calculated.

        Returns:
        R : array-like
            Reflectance of the multilayer system for each wavelength in wavelength_range.
        T : array-like
            Transmittance of the multilayer system for each wavelength in wavelength_range.
        A : array-like
            Absorption of the multilayer system for each wavelength in wavelength_range.
        """
        # Convert to array if it's a list
        wavelength_range = np.array(wavelength_range)

        # Calculate total transfer matrix for all wavelengths at once
        M = self.calculate_total_transfer_matrix(wavelength_range)

        # Calculate R
        r = M[:, 1, 0] / M[:, 0, 0]
        R = np.clip(np.abs(r)**2, 0, 1)

        # Calculate T
        n0 = np.array([self.layers[0].get_n(wavelength)
                      for wavelength in wavelength_range])
        ns = np.array([self.layers[-1].get_n(wavelength)
                      for wavelength in wavelength_range])
        t = 1 / M[:, 0, 0]
        T = np.clip(np.abs(t)**2 * (ns / n0), 0, 1)

        # Calculate A
        A = np.clip(1 - R - T, 0, 1)

        return R, T, A

    def calculate_RTA_single(self, wavelength):
        """
        Calculate the reflectance (R), transmittance (T), and absorption (A)
        for a multilayer system for a single wavelength.

        Parameters:
        wavelength: float
            Wavelength of the light to calculate R, T, A.

        Returns:
        R : float
            Reflectance of the multilayer system for this wavelength.
        T : float
            Transmittance of the multilayer system for this wavelength.
        A : float
            Absorption of the multilayer system for this wavelength.
        """
        M = self.calculate_total_transfer_matrix(wavelength)
        # Calculate R
        r = M[1, 0] / M[0, 0]
        R = np.abs(r)**2
        # np.clip is used to restrict the value within [0, 1]
        R = np.clip(R, 0, 1)

        # Calculate T
        # n of the incident medium, which is usually air
        n0 = self.layers[0].get_n(wavelength)
        ns = self.layers[-1].get_n(wavelength)  # n of the substrate
        t = 1 / M[0, 0]
        # (ns / n0) corrects for the index of the substrate
        T = np.abs(t)**2 * (ns / n0)
        T = np.clip(T, 0, 1)

        # Calculate A
        A = 1 - R - T
        A = np.clip(A, 0, 1)

        return R, T, A

    # **Plotting method**

    def plot_RTA(self, wavelength_range):
        """
        Plot the reflectance (R), transmittance (T), and absorption (A)
        for the multilayer system as a function of wavelength.
        Assume the incident light is normal.

        Parameters:
        wavelength_range : array-like
            Range of wavelengths [nm] for which R, T, A should be plotted.
        """
        # Calculate RTA
        R, T, A = self.calculate_RTA(wavelength_range)

        # Generate the plot
        plt.figure(figsize=(10, 6))

        plt.plot(wavelength_range, R, label='Reflectance')
        plt.plot(wavelength_range, T, label='Transmittance')
        plt.plot(wavelength_range, A, label='Absorption')

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Magnitude')
        plt.title('RTA vs Wavelength for Thin Film System')
        plt.legend()

    # **Methods for debugging**

    def print_matrices(self, wavelength):
        """
        Print the boundary matrices, propagation matrices, and their multiplications for the multilayer system 
        for a particular wavelength.

        Parameter:
        wavelength: float
            Wavelength of the light at which matrices are printed.
        """
        # Print the boundary matrix between air and the first layer
        B01 = self.calculate_boundary_matrix(wavelength, 0)
        print(f"B01 = {B01}")

        # Print the matrices for the rest layers, and their multiplication
        M = np.eye(2)  # create an identity matrix
        M = B01 @ M
        print(f"M = {M}")

        for index in range(1, len(self.layers) - 1):
            P = self.calculate_propagation_matrix(wavelength, index)
            print(f"P{index} = {P}")
            M = M @ P
            print(f"M = {M}")
            B = self.calculate_boundary_matrix(wavelength, index)
            print(f"B{index}{index + 1} = {B}")
            M = M @ B
            print(f"M = {M}")
            M_cal = self.calculate_total_transfer_matrix(wavelength)
            print(f"M_cal = {M_cal}")

    def print_N(self, wavelength):
        """
        Print N for each layer in the multilayer system.

        Parameter:
        wavelength: float
            Wavelength of the light at which N are calculated and printed.
        """
        for layer in self.layers:
            index = self.layers.index(layer)
            print(f"N_{index} = {layer.get_N(wavelength)}")
