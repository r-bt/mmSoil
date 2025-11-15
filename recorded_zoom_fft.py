import argparse
from src.iwr1443.radar_config import RadarConfig
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.signal import zoom_fft
from scipy.fft import fft, fftfreq
import time

def background_subtraction(frame):
    after_subtraction = np.zeros_like(frame)
    for i in range(1, frame.shape[0]):
        after_subtraction[i - 1] = frame[i] - frame[i - 1]

    return after_subtraction

def main():
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the radar config
    config = RadarConfig(args.cfg).get_params()

    c = 3e8  # speed of light - m/s
    SAMPLE_RATE = config["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = config["chirp_slope"]  # frequency slope in Hz (/s)
    SAMPLES_PER_CHIRP = config["n_samples"]  # adc number of samples per chirp

    c = 3e8  # speed of light - m/s
    SAMPLE_RATE = config["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = config["chirp_slope"]  # frequency slope in Hz (/s)
    SAMPLES_PER_CHIRP = config["n_samples"]  # adc number of samples per chirp

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(0)
    dist_plot.resize(600, 600)
    dist_plot.show()

    # Read the saved data file
    data = np.load(args.data)["data"]

    for frame in data:
        frame = background_subtraction(frame)

        # We expect reflectors to be at
        expected_reflector = 0.5 # expected distance in meters
        bounds = 0.5 # +- 50cm

        # Perform a ZoomFFT around the expected reflector
        f1 = (expected_reflector - bounds) / c * (2 * FREQ_SLOPE)
        f2 = (expected_reflector + bounds) / c * (2 * FREQ_SLOPE)

        Xz = zoom_fft(frame, [f1, f2], fs=SAMPLE_RATE, axis=1) # Across the samples per chirp axis

        fft_freqs = np.linspace(f1, f2, SAMPLES_PER_CHIRP, endpoint=False)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        X = np.average(Xz, axis=0)

        # Plot the data
        dist_plot.update(
            fft_meters,
            np.abs(X),
        )

        app.processEvents()


if __name__ == "__main__":
    main()
