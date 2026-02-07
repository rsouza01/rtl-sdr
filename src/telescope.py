#!/usr/bin/env python3

from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup the Hardware
sdr = RtlSdr()

# 2. Physics Configuration
# Sky Radio is 101.6 MHz in Eindhoven.
# We tune to 101.4 MHz so the "DC Spike" is at 101.4, 
# and the Music Station appears cleanly at +0.2 MHz (101.6).
sdr.center_freq = 101.4e6 
sdr.sample_rate = 2.048e6 

# MANUAL GAIN: Force it high (approx 30-40dB) to see the station
# 'Auto' sometimes fails on these dongles.
sdr.gain = 38.0 

print(f"Tuned to {sdr.center_freq/1e6} MHz. Looking for Station at 101.6 MHz...")

# 3. Collect Data (Take a larger snapshot for cleaner averages)
samples = sdr.read_samples(256*1024)
center_freq = sdr.center_freq
sample_rate = sdr.sample_rate
sdr.close()

# 4. Signal Processing (FFT)
# We use a window function (Blackman) to smooth out spectral leakage
# This makes the "Mountain" look sharper and reduces noise.
window = np.blackman(len(samples))
fft_out = np.fft.fftshift(np.fft.fft(samples * window))
power_db = 10 * np.log10(np.abs(fft_out)**2)

# Frequency Axis
freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))
freqs_mhz = (freqs + center_freq) / 1e6

# 5. Visualization
plt.figure(figsize=(12, 6))
plt.plot(freqs_mhz, power_db, color='#007acc', linewidth=0.8, label='Raw Data')

# MARKERS
plt.axvline(x=101.4, color='red', linestyle='--', alpha=0.5, label='DC Offset (Hardware Artifact)')
plt.axvline(x=101.6, color='green', linewidth=2, alpha=0.8, label='Sky Radio (Expected Here!)')

plt.title(f"Spectrum Analysis: Sky Radio Eindhoven (Gain: {sdr.gain}dB)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()