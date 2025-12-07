import argparse
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import zoom_fft, find_peaks
from src.iwr1443.radar_config import RadarConfig
import time
import glob
import os
import pyqtgraph as pg
import re

Nr = 4
d = 0.5 # half-wavelength spacing

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


def extract_water_amount(filename):
    """
    Extract water amount from filename.
    Format: HH-MM-DD-MM-YYYY-XXml-water.bin or HH-MM-DD-MM-YYYY-allmL-water.bin
    
    Returns:
        float or None: Water amount in ml, or None if pattern doesn't match
    """
    basename = os.path.basename(filename)
    
    # Try to match patterns like "25ml-water" or "0ml-water"
    match = re.search(r'(\d+)ml-water', basename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Check for "allml-water" pattern - we'll use a large number like 999 to represent "all"
    if re.search(r'allml-water', basename, re.IGNORECASE):
        return 999.0  # Special value for "all"
    
    return None

def extract_adc_value(filename):
    """
    Extract ADC value from filename.
    Format: adc=XXX,yyml-water.bin
    
    Returns:
        float or None: ADC value, or None if pattern doesn't match
    """
    basename = os.path.basename(filename)
    
    # Try to match patterns like "XXadc-value"
    match = re.search(r'adc=(\d+)', basename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None


def process_bin_file(filepath, params):
    """
    Process a .bin file and return the average FFT magnitude and frequencies.
    
    Returns:
        tuple: (fft_meters, avg_fft_magnitude)
    """
    c = 3e8  # speed of light - m/s
    SAMPLES_PER_CHIRP = params["n_samples"]
    SAMPLE_RATE = params["sample_rate"]
    CHIRPS_PER_FRAME = params["n_chirps"]
    FREQ_SLOPE = params["chirp_slope"]
    
    # Load the data from the .bin file
    fid = open(filepath, 'rb')
    raw_data = np.fromfile(fid, dtype='<i2')  # Little-endian int16
    fid.close()

    raw_data_reshaped = raw_data.reshape(-1, 8)  # Each lane has I and Q
    adc_data = raw_data_reshaped[:, :4] + 1j * raw_data_reshaped[:, 4:]
    data = adc_data.reshape(-1, CHIRPS_PER_FRAME, SAMPLES_PER_CHIRP, Nr)

    # Accumulate FFT magnitudes across all frames
    fft_accumulator = None
    frame_count = 0
    
    for frame in data:
        frame = np.average(frame, axis=0)  # Now frame is (n_samples, n_rx)
        frame = frame.T  # Now frame is (n_rx, n_samples)

        # First apply beamsteering to only get reflectors at 0 degrees
        w = w_mvdr(0, frame)  # get weights for 0 degrees
        frame_weighted = w.conj().T @ frame  # apply weights

        Xz = fft(frame_weighted, axis=1)
        Xz = Xz[0]
        
        # Accumulate magnitude
        if fft_accumulator is None:
            fft_accumulator = np.abs(Xz)
        else:
            fft_accumulator += np.abs(Xz)
        frame_count += 1

    # Average the accumulated FFT
    avg_fft = fft_accumulator / frame_count
    
    # Calculate frequencies and distances
    fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
    fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)
    
    # Return only positive frequencies
    return fft_meters[:SAMPLES_PER_CHIRP // 2], avg_fft[:SAMPLES_PER_CHIRP // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing .bin files")
    args = parser.parse_args()

    # Find all .bin files in the data directory
    bin_files = glob.glob(os.path.join(args.data_dir, "*.bin"))
    
    if not bin_files:
        print(f"No .bin files found in {args.data_dir}")
        return
    
    print(f"Found {len(bin_files)} .bin files:")
    for f in bin_files:
        print(f"  - {os.path.basename(f)}")

    # Initialize the radar config
    params = RadarConfig(args.cfg).get_params()

    # Initialize the GUI with two plots side by side
    app = QtWidgets.QApplication(sys.argv)
    
    # Create main window with layout for two plots
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("FFT Comparison - All .bin Files")
    
    central_widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout()
    central_widget.setLayout(layout)
    main_window.setCentralWidget(central_widget)
    
    # FFT plot (left)
    fft_plot_widget = pg.PlotWidget()
    fft_plot_widget.setLabel("bottom", "Distance (m)")
    fft_plot_widget.setLabel("left", "Intensity")
    fft_plot_widget.addLegend()
    layout.addWidget(fft_plot_widget)
    
    # Max amplitude vs water amount plot (right)
    amp_plot_widget = pg.PlotWidget()
    amp_plot_widget.setLabel("bottom", "Water Amount (ml)")
    amp_plot_widget.setLabel("left", "Max Amplitude")
    amp_plot_widget.setTitle("Max Amplitude vs Water Amount")
    layout.addWidget(amp_plot_widget)
    
    main_window.resize(1600, 600)
    
    # Define distinct colors for each file
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 150, 255),    # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (128, 255, 0),    # Lime
        (255, 0, 128),    # Pink
    ]
    
    # Store data for water amount vs max amplitude plot
    water_data = []
    
    # Process each file and plot
    for idx, bin_file in enumerate(bin_files):
        print(f"\nProcessing {os.path.basename(bin_file)}...")
        
        fft_meters, avg_fft_mag = process_bin_file(bin_file, params)
        
        # Create a label from the filename (remove .bin extension)
        label = os.path.splitext(os.path.basename(bin_file))[0]
        
        # Choose color
        color = colors[idx % len(colors)]
        
        # Plot the FFT on left plot
        fft_plot_widget.plot(
            fft_meters,
            avg_fft_mag,
            pen=pg.mkPen(color=color, width=2),
            name=label
        )
        
        # Get max amplitude
        max_amp = np.max(avg_fft_mag)
        max_dist = fft_meters[np.argmax(avg_fft_mag)]
        
        print(f"  Max intensity: {max_amp:.2f} at {max_dist:.3f}m")
        
        # Extract water amount
        water_ml = extract_water_amount(bin_file)
        if water_ml is not None:
            print(f"  Water amount: {water_ml} ml")
        
        adc_value = extract_adc_value(bin_file)
        if adc_value is not None:
            print(f"  ADC value: {adc_value}")
        
        water_data.append({
            'water_ml': water_ml,
            'adc_value': adc_value,
            'max_amp': max_amp,
            'label': label,
            'color': color
        })
    
    # Sort water data by ADC value
    water_data.sort(key=lambda x: x['adc_value'])
    
    # Plot max amplitude vs ADC value
    if water_data:
        adc_values = [d['adc_value'] for d in water_data]
        max_amplitudes = [d['max_amp'] for d in water_data]
        
        # Plot with scatter points and connecting line
        amp_plot_widget.plot(
            adc_values,
            max_amplitudes,
            pen=pg.mkPen(color=(100, 100, 255), width=2),
            symbol='o',
            symbolSize=10,
            symbolBrush=(100, 100, 255)
        )
        
        # Add text labels for each point
        for d in water_data:
            text = pg.TextItem(
                text=f"ADC={d['adc_value']}" if d['adc_value'] is not None else "ADC=N/A",
                anchor=(0.5, 1.5),
                color=(255, 255, 255)
            )
            text.setPos(d['adc_value'], d['max_amp'])
            amp_plot_widget.addItem(text)
    
    # Set reasonable axis ranges
    fft_plot_widget.setXRange(0, 0.75)
    
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
