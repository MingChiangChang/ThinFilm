from ThinFilmClasses import ThinFilmLayer, ThinFilmSystem
from DataProcessing import process_data, data_smoothing, plot_reflectance

# Provide the path of the data, lower bound of the wavelength (left), and upper bound of the wavelength (right).
path = r'test_data\spectra\+02_+05.csv'
left = 380
right = 830
data = process_data(path, left, right)
plot_reflectance(data=data, smoothed_data=True)
