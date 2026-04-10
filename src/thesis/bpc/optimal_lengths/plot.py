from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from pathlib import Path
import json

def calc_stats_single(file_path):
    population = []
    bpcs = []
    with open(file_path, 'r') as f:
        for line in f:
            curr = json.loads(line)
            for key, value in curr.items():
                population.append(value['n'])
                bpcs.append(value['avg_bpc'])
    population = np.array(population, dtype=int)
    bpcs = np.array(bpcs, dtype=float)
    
    # Get the distribution
    values = np.bincount(population)
    values = values/sum(values)

    # Calculate stats
    peak = np.argmax(values)
    variance = bpcs.var()
    std_dev = bpcs.std()
    mean = population.mean()
    
    return population

def avg_bpc_per_length(file_path):
    bpc_dict = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            curr = json.loads(line)
            for key, value in curr.items():
                bpc_dict[value['n']].append(value['avg_bpc'])
    
    # Extract only existing lengths and sort them
    lengths = sorted(bpc_dict.keys())
    
    # Calculate the mean only for lengths that exist
    avg_bpc = [np.mean(bpc_dict[l]) for l in lengths]

    return lengths, avg_bpc
'''
def calc_stats_multi(dir_path):
    folder = Path(dir_path)

    # Dictionary containing populations for each setting of languages lengths
    all_populations = {}
    for file in folder.iterdir():
        if file.is_file():
            name=str(file).split('/')[-1]
            if '_5_True' in name:
                all_populations[str(file).split('/')[-1]] = calc_stats_single(file)
    
    plt.figure()

    all_data = np.concatenate(list(all_populations.values()))
    x_smooth = np.linspace(min(all_data), min(max(all_data), 300), 500)

    for label, population in all_populations.items():
        kde = gaussian_kde(population)
        y_smooth = kde(x_smooth)
        
        plt.plot(x_smooth, y_smooth, label=label)

    plt.title("KDE Smoothed PDFs")
    plt.xlim(0, 300) 
    plt.legend()
    plt.show()
'''
def plot_bpc_vs_n(file):
    population = calc_stats_single(file)

    x, y = avg_bpc_per_length(file)
    y_smoothed = savgol_filter(y, window_length=11, polyorder=3)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # AXIS 1: The BPC Data (Left side)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of data points')
    ax1.set_ylabel('Average BPC', color=color1)

    ax1.plot(x, y_smoothed, color=color1, linewidth=2, label='Avg. BPC')
    ax1.tick_params(axis='y', labelcolor=color1)

    # AXIS 2: The Distribution/PDF (Right side)
    kde = gaussian_kde(population, bw_method='scott')

    # 2. Generate a smooth, high-resolution X-axis spanning your min and max lengths
    x_pdf_smooth = np.linspace(population.min(), population.max(), 500)

    # 3. Evaluate the KDE curve over those X values
    smoothed_pdf = kde(x_pdf_smooth)

    # --- AXIS 2: The Distribution/PDF (Right side) ---
    ax2 = ax1.twinx()  

    color2 = 'tab:red'
    ax2.set_ylabel('Population Density (KDE)', color=color2)

    # Plot the smooth KDE curve
    ax2.plot(x_pdf_smooth, smoothed_pdf, color=color2, linestyle='-', linewidth=2, alpha=0.9, label='KDE')

    ax2.tick_params(axis='y', labelcolor=color2)

    # --- Clean up and Show ---
    max_x = len(y_smoothed)
    ax1.set_xlim(0, max_x)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title("Average BPC and Population Distribution vs Number of Samples")
    fig.tight_layout()
    plt.show()