import argparse
from src.iwr1443.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.signal import zoom_fft

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

        # We expect reflectors to be at
        expected_reflector = 1
        bounds = 0.1 # +- 2cm

        # Perform a ZoomFFT around the expected reflector
        f1 = (expected_reflector - bounds) / c * (2 * FREQ_SLOPE)
        f2 = (expected_reflector + bounds) / c * (2 * FREQ_SLOPE)
        
        m = 4096                     # high-res output (zoom)
        Xz = zoom_fft(frame, [f1, f2], m=m, fs=SAMPLE_RATE, axis=1) # Across the samples per chirp axis

        fft_freqs = np.linspace(f1, f2, m, endpoint=False)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Plot the data
        dist_plot.update(
            fft_meters,
            np.abs(Xz[0, :, :]),
        )
        
        app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
