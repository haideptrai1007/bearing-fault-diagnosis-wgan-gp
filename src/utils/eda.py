import os
os.chdir("..")
from src.utils.utils import sliding_window

import matplotlib.pyplot as plt
import numpy as np
import random
import math

def plot_raw_signal(fault_list, label_list, title, fs=12000, segment=False):
    n = len(fault_list)
    
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n == 1:
        axes = np.array([axes])
        
    axes = axes.flatten()

    for i in range(n):
        if segment:
            fault_windows = sliding_window(fault_list[i])
            fault = fault_windows[random.randint(0, len(fault_windows) - 1)]
        else:
            fault = fault_list[i]
            
        label = label_list[i]
        
        axes[i].plot(np.arange(len(fault)) / fs, fault, linewidth=0.5)
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f"{label} Vibration Signal")
        axes[i].grid(True)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()