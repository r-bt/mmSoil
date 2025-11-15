import argparse
from src.iwr1443.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.range_angle_plot import RangeAngleHeatmap
import sys
from scipy.fft import fft, fftfreq

Nr = 4
d = 0.5 # half-wavelength spacing

def steering_vector(theta, Nr, d=0.5):
    """
    ULA steering vector in units of wavelength
    theta: angle in radians
    Nr: number of RX antennas
    d: spacing in wavelengths (default 0.5)
    """
    n = np.arange(Nr)
    return np.exp(-1j * 2 * np.pi * d * n * np.sin(theta))[:, None]  # (Nr,1)

def mvdr_spectrum(X, angles_deg, d=0.5):
    """
    Compute MVDR power spectrum for a single range bin
    X: (Nr, n_snapshots) array for this bin
    angles_deg: array of angles to scan
    """
    Nr = X.shape[0]
    R = X @ X.conj().T / X.shape[1]        # spatial covariance
    R += 1e-3 * np.eye(Nr)                 # diagonal loading
    Rinv = np.linalg.pinv(R)
    
    P = np.zeros(len(angles_deg))
    for i, th in enumerate(angles_deg):
        a = steering_vector(th, Nr, d)
        denom = np.real(a.conj().T @ Rinv @ a)
        P[i] = 1.0 / denom
    return 10 * np.log10(P / P.max() + 1e-12)

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

    heatmap = RangeAngleHeatmap(range_res=params["range_res"], angle_bins=params["n_rx"])
    heatmap.resize(600, 600)
    heatmap.show()

    def update_frame(msg):
        frame = msg.get("data", None)
        if frame is None:
            return
        
        # Calculate the range fft
        range_fft = np.fft.fft(frame, axis=1)           # (n_chirps, n_samp, n_rx)
        range_fft = range_fft[:, :SAMPLES_PER_CHIRP//2, :]         # positive half
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        ranges = fft_freqs[: SAMPLES_PER_CHIRP // 2] * c / (2 * FREQ_SLOPE)

        # Get the AoA
        theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees

        RA_map = np.zeros((len(ranges), len(theta_scan)))
    
        for r in range(len(ranges)):
            X = range_fft[:, r, :].T  # shape (n_rx, n_chirps)
            RA_map[r, :] = mvdr_spectrum(X, theta_scan  , d)

        # Plot the data
        heatmap.update(RA_map)
        app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
