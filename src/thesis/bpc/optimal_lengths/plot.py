from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy import integrate

from pathlib import Path
import json

def calculate_kde_overlap(kde_true, kde_false, x_range):
    # 1. Evaluate both PDFs over the same range
    pdf_true = kde_true(x_range)
    pdf_false = kde_false(x_range)

    # 2. Find the minimum value at every point along the x-axis
    # This represents the "bottom" curve of the overlapping region
    overlap_pdf = np.minimum(pdf_true, pdf_false)

    # 3. Integrate using the trapezoidal rule
    # np.trapz calculates the area under the 'overlap_pdf' curve
    overlap_area = integrate.trapezoid(overlap_pdf, x_range)

    return overlap_area

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
    
    return population, bpcs

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
    population, _ = calc_stats_single(file)

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

'''
Plots the distribution of BPC for true and false languages
'''
def plot_true_vs_false(true_file, false_file):
    # Assuming calc_stats_single returns (mean, list_of_bpcs)
    _, true_bpcs = calc_stats_single(true_file)
    _, false_bpcs = calc_stats_single(false_file)
    
    true_bpcs = np.array(true_bpcs)
    false_bpcs = np.array(false_bpcs)

    # 1. Create KDEs for both populations
    kde_true = gaussian_kde(true_bpcs, bw_method='scott')
    kde_false = gaussian_kde(false_bpcs, bw_method='scott')

    # 2. Generate a smooth X-axis spanning the range of all data
    all_data = np.concatenate([true_bpcs, false_bpcs])
    x_range = np.linspace(all_data.min(), all_data.max(), 1000)

    # 3. Evaluate the KDE curves
    pdf_true = kde_true(x_range)
    pdf_false = kde_false(x_range)

    # 4. Find the Threshold (Intersection Point)
    # This finds where the 'unseen' BPC starts becoming more likely than 'seen' BPC
    idx = np.argwhere(np.diff(np.sign(pdf_true - pdf_false))).flatten()
    threshold = x_range[idx[0]] if len(idx) > 0 else np.mean(all_data)

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, pdf_true, label='In Training (Seen)', color='teal', lw=2)
    plt.fill_between(x_range, pdf_true, alpha=0.2, color='teal')
    
    plt.plot(x_range, pdf_false, label='Not in Training (Unseen)', color='crimson', lw=2)
    plt.fill_between(x_range, pdf_false, alpha=0.2, color='crimson')

    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.3f}')
    
    plt.title('BPC Distribution: Seen vs. Unseen Languages')
    plt.xlabel('Bits Per Character (BPC)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    print(f"Suggested BPC Threshold: {threshold:.4f}")

    x_range = np.linspace(all_data.min(), all_data.max(), 1000)
    area = calculate_kde_overlap(kde_true, kde_false, x_range)
    print(f"Overlap Area: {area:.4f}")
    return threshold