#!/usr/bin/env python3

# Play results with: `afplay sky_radio.wav` (Linux) or open in Audacity/Media Player.

import numpy as np
from rtlsdr import RtlSdr
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
radio_freq_mhz = 101.58     # Sky Radio (Eindhoven)
station_freq = radio_freq_mhz * 1e6
center_freq = (radio_freq_mhz - 0.2) * 1e6    # Tune slightly off-center to avoid DC spike
sample_rate = 240000     # 240k samples/sec (Integer multiple of 48k audio)
duration = 10            # Seconds to record

# 1. HARDWARE: Capture the Photons
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = 55  # Manual Gain (Adjust if static is too loud)

print(f"Recording {duration}s of raw IQ data from {station_freq/1e6} MHz...")
# Read samples
num_samples = int(sample_rate * duration)
samples = sdr.read_samples(num_samples)
sdr.close()

# 2. PHYSICS: Shift the Frequency
# We are tuned to 101.4, Station is at 101.6.
# We must shift the signal down by 200kHz to center it at 0Hz (Baseband).
offset_hz = station_freq - center_freq # +200,000 Hz
t = np.arange(len(samples)) / sample_rate
# Multiply by complex exponential to shift frequency
samples_shifted = samples * np.exp(-1j * 2 * np.pi * offset_hz * t)

# 3. DEMODULATION: The Math of FM
# FM encodes audio in the *rate of change* of the phase angle.
# Audio ~ d(Phase)/dt
# We use np.angle to get phase, np.unwrap to fix the 2pi jumps, and np.diff for derivative.
phase = np.unwrap(np.angle(samples_shifted))
audio_raw = np.diff(phase)

# 4. DSP: Decimation & De-emphasis
# Downsample from 240kHz (Radio rate) to 48kHz (Audio rate)
audio_down = signal.decimate(audio_raw, 5) # 240k / 5 = 48k

# Normalize to 16-bit integer range for WAV format
audio_norm = np.int16(audio_down/np.max(np.abs(audio_down)) * 32767)

# 5. OUTPUT: Save to Disk
filename = "sky_radio.wav"
wavfile.write(filename, 48000, audio_norm)
print(f"Done! Saved to {filename}")