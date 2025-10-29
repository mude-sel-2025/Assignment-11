# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def task1_load_and_plot(filename):
    npzfile = np.load(filename)
    
    # Check variable names
    print("Available variables:", npzfile.files)
    
    # Adjust variable name as per your saved file (free_vibration.npz)
    t = npzfile['t']; x = npzfile['x']
    
    plt.figure(figsize=(10,4))
    # YOUR CODE HERE: hint : plt.plot plots array x vs array y as plot(x,y)
    plt.plot() 
    plt.xlabel('Time (s)')
    plt.ylabel('Measured Displacement (m)')
    plt.title('Measured Free Vibration Response')
    plt.grid(True)

    return t, x


def task2_compute_fft(t, x):
    '''YOUR CODE HERE: In Python, you can use np.fft.fft() to get X(f) and 
    np.fft.fftfreq() to get f '''
    dt = t[1]-t[0]
    N = len(x)
    
    # YOUR CODE HERE: Compute fft of x(t) and corresponding frequencies. Hint: lookup the inputs you need to provide
    Xf = np.fft.fft()
    f = np.fft.fftfreq()
    
    # Plotting just the magnitudes of complexes
    Xf_amp = np.abs(Xf)

    mask = f > 0
    f = f[mask]; Xf_amp = Xf_amp[mask]
    
    plt.figure()
    plt.plot(f, Xf_amp)
    plt.xlim((0, 100))
    plt.xlabel('Frequency [Hz]'); plt.ylabel('|X(f)|')
    plt.title('Frequency Spectrum')

    fn = f[np.argmax(Xf_amp)]
    print(f"Estimated natural frequency = {fn:.2f} Hz")
    
    return fn


def task3_filtering(t, x, fn):
    # Band-pass filter around dominant frequency ±5 Hz
    Xf = np.fft.fft(x) 
    freq_full = np.fft.fftfreq(len(t), d=t[1]-t[0])
    
    # YOUR CODE HERE: Find indices of freq_full > f_noise_lower and freq_full < f_noise_higher = 0
    indices = 
    # Set the frequency content = 0 for these frequencies
    Xf[indices] = 0
    # Take inverse fourier transform
    x_filtered = np.real(np.fft.ifft(Xf))
    
    plt.figure()
    plt.subplot(2,1,1); plt.plot(t, x); plt.title('Original Signal')
    plt.subplot(2,1,2); plt.plot(t, x_filtered); plt.title('Filtered Signal')
    plt.tight_layout(); 
    
    return x_filtered
    
    
def task4_estimate_damping(t, x_filtered):
    
    peaks, _ = find_peaks(x_filtered)
    A = x_filtered[peaks]
    ''' print first twenty peaks. Hint: Note that indices of distinct positive peaks 
    from here and number of cycles in between them '''
    print(f"Picked peaks:{A[0:20]}")
    # YOUR CODE HERE: Compute logarithmic decrement using first and (N+1)th peak
    N =   # number of cycles apart
    delta =     # log decrement
    zeta =      # damping ratio

    print(f"Logarithmic Decrement = {delta:.3f}")
    print(f"Damping Ratio = {zeta*100:.2f}%")
    
    plt.figure()
    plt.plot(t, x_filtered)
    plt.plot(t[peaks], A, 'ro')
    plt.title('Filtered Response with Peaks Marked')
    plt.xlabel('Time [s]')

    return zeta

def task5_aliasing(t, x_filtered):
   
    dt = t[1]-t[0]
    print(f'Original Sampling Rate: {dt}, Original Sampling Frequency: {1/dt}')
    
    # YOUR CODE HERE: Set down-sampling rate
    N =                 # Downsampling rate
   
    # YOUR CODE HERE: Compute sub-sampled displacement and time signals
    x2 = 
    t2 = 
    dt = t2[1]-t2[0]
    print(f'Sub-sampled Sampling Rate: {dt}, Sub-sampled Sampling Frequency: {1/dt}')
    
    # Taking FFT of Sub-sampled signal
    Xf = np.fft.fft(x2)
    f = np.fft.fftfreq(len(t2), dt)
    Xf_amp = np.abs(Xf)
    mask = f > 0
    f = f[mask]; Xf_amp = Xf_amp[mask]
    
    # Plotting Sub-sampled Signal and its FFT
    plt.figure()
    plt.subplot(2,1,1); plt.plot(t2, x2); plt.ylabel('x(t)')
    plt.title('Sub-sampled displacement response')
    plt.subplot(2,1,2)
    plt.plot(f, Xf_amp); plt.ylabel('|X(f)|')
    plt.title('Frequency Spectrum of sub-sampled signal')
    plt.tight_layout(); 
    
    
if __name__ == "__main__":
    filename = 'free_vibration.npz'
    
    # TASK 1: Plot and Visualize the Free Vibration Response
    t, x = task1_load_and_plot(filename)
    
    # TASK 2: Compute FFT and Visualize the frequency contents
    fn = task2_compute_fft(t, x)
    
    # TASK 3: Filter out unwanted noise from the displacement response
    x_filtered = task3_filtering(t, x, fn)
    
    # TASK 4: Find damping using filtered response
    zeta = task4_estimate_damping(t, x_filtered)
        
    # TASK 5: Effect of low sampling rate
    task5_aliasing(t, x_filtered)
    
    plt.show()
