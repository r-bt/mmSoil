import argparse
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import zoom_fft, find_peaks
from src.iwr1443.radar_config import RadarConfig

Nr = 4
d = 0.5 # half-wavelength spacing

TOP_TO_FOIL_DISTANCE = 0.11 # 10cm

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
    parser.add_argument("--data", type=str, required=True, help="Path to the .bin file")
    args = parser.parse_args()

    # Initalize the radar config
    params = RadarConfig(args.cfg).get_params()

    c = 3e8  # speed of light - m/s
    SAMPLES_PER_CHIRP = params["n_samples"]  # adc number of samples per chirp
    SAMPLE_RATE = params["sample_rate"]  # digout sample rate in Hz
    
    FREQ_SLOPE = params["chirp_slope"]  # frequency slope in Hz (/s)

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(params["range_res"])
    dist_plot.resize(600, 600)
    dist_plot.show()

    # Load the data from the .bin file
    fid = open(args.data, 'rb')
    raw_data = np.fromfile(fid, dtype='<i2')  # Little-endian int16
    fid.close()

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
        # expected_reflector = 0.4 # expected distance in meters
        # bounds = 0.1 # +- 50cm
        # d1 = 0
        # d2 = 2

        # # Perform a ZoomFFT around the expected reflector
        # f1 = (d1) / c * (2 * FREQ_SLOPE)
        # f2 = (d2) / c * (2 * FREQ_SLOPE)

        # Xz = zoom_fft(frame_weighted, [f1, f2], fs=SAMPLE_RATE, axis=1) # Across the samples per chirp axis
        # Xz = Xz[0]

        Xz = fft(frame_weighted, axis=1)
        Xz = Xz[0]

        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        print(Xz.shape)

        # # Find the range bin with the maximum value
        # peaks, props = find_peaks(np.abs(Xz), height=30, prominence=10, distance=3)

        # if not np.any(peaks):
        #     return
        
        # heights = props["peak_heights"]

        # highest_peak = peaks[np.argmax(heights)]

        # # Filter to peaks to the right
        # right_mask = peaks > highest_peak
        
        # if not np.any(right_mask):
        #     dist_plot.update(
        #         fft_meters,
        #         np.abs(Xz),
        #         vertical_lines=[fft_meters[highest_peak]]
        #     )
        #     app.processEvents()
        #     return
        
        # right_peaks = peaks[right_mask]
        # right_heights = heights[right_mask]

        # next_peak = right_peaks[np.argmax(right_heights)]
        
        # top_2_peaks = [highest_peak, next_peak]

        # peaks_in_meters = [fft_meters[peak] for peak in top_2_peaks]

        # # Convert the peaks into a time difference
        # times = [fft_freqs[idx] * 1/(2 * FREQ_SLOPE) for idx in np.sort(top_2_peaks)]
        # delta_T = times[1] - times[0]

        # print(delta_T)

        # # Calculate the dielectric permittivity
        # epsilon = ((c * delta_T) / TOP_TO_FOIL_DISTANCE) ** 2

        # # Calculate the VWC
        # VWC = -5.3*10**(-2) + 2.92*10**(-2)*epsilon - 5.5*10**(-4)*epsilon**2 + 4.3*10**(-6)*epsilon**3

        # print(VWC)

        # Plot the data
        dist_plot.update(
            fft_meters[: SAMPLES_PER_CHIRP // 2],
            np.abs(Xz[: SAMPLES_PER_CHIRP // 2]),
            vertical_lines=[]
        )

        app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
