import argparse
from src.iwr1443.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.range_angle_plot import RangeAngleHeatmap
import sys
from scipy.fft import fft, fftfreq


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
        
        padding_size=[128, 64, 32]

        if padding_size is None:
            padding_size = data.shape
        
        # Apply a hanning window to reduce spectral leakage
        window = np.hanning(frame.shape[0])[:, None] * np.hanning(frame.shape[1])[None, :]

        channels = frame.shape[2]

        for i in range(channels):
            frame[:, :, i] = frame[:, :, i] * window

        # Apply a 2D fft to get range-doppler map
        data = np.fft.fft2(data, s=[padding_size[0], padding_size[1]], axes=[0, 1])

        # Get range angle by applying fft along the rx axis
        rai_abs = np.fft.fft(data, n=padding_size[2], axis=2)
        rai_abs = np.fft.fftshift(np.abs(rai_abs), axes=2)
        rai_abs = np.flip(rai_abs, axis=1)

        # Average over the chirps
        rai_avg = np.mean(rai_abs, axis=0)

        print("shape: ", rai_avg.shape)

        # frame = background_subtraction(frame)

        # # Get the fft of the data
        # signal = np.mean(frame, axis=0)

        # fft_result = fft(signal, axis=0)
        # fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        # fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # # Plot the data
        # dist_plot.update(
        #     fft_meters[: SAMPLES_PER_CHIRP // 2],
        #     np.abs(fft_result[: SAMPLES_PER_CHIRP // 2, :]),
        # )

        # app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
