#!/usr/bin/env python3


from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
import time

# --- CONFIGURATION ---
center_freq = 101.4e6    # Still aiming near Sky Radio
sample_rate = 2.048e6    # 2 MHz view
num_scans = 50           # We will stack 50 images on top of each other
n_samples = 256*1024     # Size of each snapshot

# 1. SETUP
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = 35

# Prepare an empty array to hold our accumulated photons
# We use a standard FFT size based on the sample chunk
fft_size = n_samples 
power_accumulator = np.zeros(fft_size)

print(f"Starting Integration: {num_scans} scans...")
start_time = time.time()

# 2. THE LOOP (The 'Exposure' Time)
for i in range(num_scans):
    # A. Grab raw voltage
    samples = sdr.read_samples(n_samples)
    
    # B. Convert to Power (Linear, not dB yet!)
    # We use a Blackman window to smooth the edges
    window = np.blackman(len(samples))
    fft_snap = np.fft.fftshift(np.fft.fft(samples * window))
    power_snap = np.abs(fft_snap)**2
    
    # C. Accumulate
    power_accumulator += power_snap
    
    # Simple progress bar
    print(f"Scan {i+1}/{num_scans}", end='\r')

sdr.close()
print(f"\nIntegration complete in {time.time() - start_time:.2f}s")

# 3. AVERAGE & CONVERT TO DECIBELS
# Average Power = Total Energy / Count
avg_power = power_accumulator / num_scans
# Convert to dB
power_db = 10 * np.log10(avg_power)

# Frequency Axis
freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, 1/sample_rate))
freqs_mhz = (freqs + center_freq) / 1e6

# 4. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(freqs_mhz, power_db, color='orange', linewidth=1, label=f'Integrated ({num_scans} scans)')

# Markers
plt.axvline(x=101.6, color='green', linestyle=':', label='Sky Radio')

plt.title(f"Deep Field Integration ({num_scans}x Averaging)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power Spectral Density (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()