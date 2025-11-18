import argparse
from src.iwr1443.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import zoom_fft

Nr = 4
d = 0.5 # half-wavelength spacing

def background_subtraction(frame):
    after_subtraction = np.zeros_like(frame)
    for i in range(1, frame.shape[0]):
        after_subtraction[i - 1] = frame[i] - frame[i - 1]

    return after_subtraction

def w_mvdr(theta, X):
    """
    Get the MVDR weights for angle theta
    theta: angle in radians
    X: (Nr, n_samples) array for this bin
    """
    s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector in the desired direction theta
    s = s.reshape(-1,1) # make into a column vector (size
    R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # pseudo-inverse tends to work better/f
    w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # Initalize the radar
    print("initinalizing radar")
    radar = Radar(args.cfg, host_ip="192.168.33.42")
    print("inited radar")

    params = radar.params

    c = 3e8  # speed of light - m/s
    SAMPLES_PER_CHIRP = params["n_samples"]  # adc number of samples per chirp
    SAMPLE_RATE = params["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = params["chirp_slope"]  # frequency slope in Hz (/s)

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(params["range_res"])
    dist_plot.resize(600, 600)
    dist_plot.show()

    def update_frame(msg):
        frame = msg.get("data", None)
        if frame is None:
            return
        
        # Average over the chirps
        frame = np.average(frame, axis=0)  # Now frame is (n_samples, n_rx)
        frame = frame.T  # Now frame is (n_rx, n_samples)

        # First apply beamsteering to only get reflectors at 0 degrees
        w = w_mvdr(0, frame) # get weights for 0 degrees

        frame_weighted = w.conj().T @ frame  # apply weights

        # We expect reflectors to be at
        expected_reflector = 0.5 # expected distance in meters
        bounds = 0.5 # +- 50cm

        # Perform a ZoomFFT around the expected reflector
        f1 = (expected_reflector - bounds) / c * (2 * FREQ_SLOPE)
        f2 = (expected_reflector + bounds) / c * (2 * FREQ_SLOPE)

        Xz = zoom_fft(frame_weighted, [f1, f2], fs=SAMPLE_RATE, axis=1) # Across the samples per chirp axis

        fft_freqs = np.linspace(f1, f2, SAMPLES_PER_CHIRP, endpoint=False)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Find the range bin with the maximum value
        max_idx = np.argmax(np.abs(Xz.T), axis=0)[0]

        # Plot the data
        dist_plot.update(
            fft_meters,
            np.abs(Xz.T),
            vertical_lines=[fft_meters[max_idx]],
        )

        app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
