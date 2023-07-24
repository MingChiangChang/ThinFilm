import numpy as np
from scipy.optimize import curve_fit


class CauchyFit:
    def __init__(self, x, y):
        self.popt, _ = curve_fit(self.cauchy_eqn, x, y)

    def cauchy_eqn(self, wavelength, A, B, C):
        wavelength_squared = wavelength ** 2
        wavelength_fourth = wavelength ** 4
        n = A + B / wavelength_squared + C / wavelength_fourth
        return n

    def __call__(self, x):
        return self.cauchy_eqn(x, *self.popt)
