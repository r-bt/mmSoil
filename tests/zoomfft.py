import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import zoom_fft

fs = 1000
T = 2
t = np.linspace(0, T, int(fs*T), endpoint=False)

x = (
    np.sin(2*np.pi*1*t) +
    np.sin(2*np.pi*6*t) +
    np.sin(2*np.pi*10*t) +
    np.sin(2*np.pi*12*t) +
    np.sin(2*np.pi*100*t)
)

# Add some noise
x += 0.5 * np.random.randn(len(t))

# zoom settings
f1, f2 = 5, 15               # 10 Hz ± 5 Hz
m = 4096                     # high-res output (zoom)

Xz = zoom_fft(x, [f1, f2], m=m, fs=fs)
freqs = np.linspace(f1, f2, m, endpoint=False)

# Also for comparison calculate the normal FFT but truncated to the same region
X = np.fft.fft(x)
freqs_full = np.fft.fftfreq(len(x), 1/fs)

plt.figure()
plt.subplot(2,1,1)
plt.title("Normal FFT")
plt.plot(freqs_full, np.abs(X))
plt.xlim(f1, f2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.subplot(2,1,2)
plt.title("Zoom FFT")
plt.plot(freqs, np.abs(Xz))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()