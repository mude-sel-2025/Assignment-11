# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import lstsq

def task1_load_and_plot(filename):
    data = np.load(filename)
    
    t, y = data['t'], data['y']
    
    plt.figure(figsize=(10,4))
    # YOUR CODE HERE: plt.plot plots array x vs array y as plot(x,y)
    plt.plot()
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Measured Ground Motion')
    plt.grid(True)
    
    return t, y

def task2_construct_A_mat(t, offset_time):
    
    # Step 1: Form column vectors
    ones_col =               # column for intercept y0
    time_col =                             # column for linear trend r
    offset_col = np.zeros(len(t))           # column for offset (step function)
    
    # Step 2: Activate offset column when t >= offset_time
    # YOUR CODE HERE: Set the offset = 1 after offset time & = 0 before offset time
    for i in range(len(t)):
        if t[i] >= offset_time:
            offset[i] = 
        else:
            offset[i] = 
    
    # Step 3: Combine columns manually to form A
    A = np.column_stack((ones_col, time_col, offset_col))
    
    return A
    
def task3_estimate_trend_offset(t, y, A, offset_time=10.0):
    
    # YOUR CODE HERE: Find the Xhat using lstsq(). Hint: Look at the syntax of lstsq()
    x_hat, _, _, _ = lstsq()
    y_hat = A @ x_hat
    y0, r, o = x_hat
    
    print(f"Estimated intercept y0 = {y0:.2f}")
    print(f"Estimated trend r = {r:.2f}")
    print(f"Estimated offset o = {o:.2f}")
    
    plt.figure(figsize=(10,4))
    plt.plot(t, y, label='Filtered Signal')
    plt.plot(t, y_hat, 'r--', label='Model Fit')
    plt.legend(); plt.grid(True)
    plt.title('Estimated Trend and Offset')
    
    return y0, r, o

def task4_remove_trend_offset(t, y, y0, r, o, offset_time=10.0):
    
    u = (t >= offset_time).astype(float)
    
    # YOUR CODE HERE: Estimate the part of ground motion to be removed, i.e. y0hat + rhat*t + ohat*u
    trend_offset = 
    
    plt.figure()
    plt.plot(t, trend_offset)
    
    # YOUR CODE HERE: Find the "clean" ground motion, i.e. y - (y0hat + rhat*t + ohat*u) 
    y_clean = 
    
    plt.figure(figsize=(10,4))
    plt.plot(t, y, label='Filtered Signal', alpha=0.6)
    plt.plot(t, y_clean, 'r', label='Final Clean Signal', linewidth=1.2)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Final Clean Ground Motion')
    plt.legend(); plt.grid(True)
    
    return y_clean

def task5_visualize_fft(t, y_clean):
    # YOUR CODE HERE: In Python, you can use np.fft.fft() to get Y(f) and 
    # np.fft.fftfreq() to get f
    Yf = np.fft.fft()
    f = np.fft.fftfreq()
    Yf_amp = np.abs(Yf)

    mask = f > 0
    f = f[mask]; Yf_amp = Yf_amp[mask]
    
    plt.figure()
    plt.plot(f, Yf_amp)
    plt.xlim((0, 50))
    plt.xlabel('Frequency [Hz]'); plt.ylabel('|X(f)|')
    plt.title('Frequency Spectrum')


if __name__ == "__main__":
    
    filename = "ground_motion.npz"
    # TASK 1: Load and visualize the ground motion
    t, y = task1_load_and_plot(filename)
    
    # TASK 2: Construct A matrix for AX = Y
    # YOUR CODE HERE: Fill in offset_time
    # offset_time = 
    # A = task2_construct_A_mat(t, offset_time)
    
    # TASK 3: Estimate rates, bias and offset
    # y0, r, o = task3_estimate_trend_offset(t, y, A, offset_time)
    
    # TASK 4: Recover corrected ground motion
    # y_clean = task4_remove_trend_offset(t, y, y0, r, o)
    
    # TASK 5: Visualize frequency content
    # task5_visualize_fft(t, y_clean)
    
    plt.show()
