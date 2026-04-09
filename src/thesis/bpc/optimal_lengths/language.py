import thesis.paths as paths

import math

from collections import defaultdict

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
import torch

from pathlib import Path

import json

import datasets
from huggingface_hub import ModelCard

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

'''
Takes in a model for Masked LM and returns the loss obtained by
masking each token one-by-one.

Inputs
--------
tokenizer       - the tokenizer (actual tokenizer, not path)
model           - the model (actual model, not path)
text: str       - text input; not tokenized
batch_size: int - size to which divide up the matrix resulting from
                  creating a len(sentence)xlen(sentence) matrix
'''
def encoder_full_loss(tokenizer, model, text, batch_size=16):
    model.eval()
    device = model.device

    # Check if the model needs to do truncation (some don't have a max length set)
    # (the 100000 is arbitrary but i don't think there's any longer data points in the dataset)
    kwargs = {"return_tensors": "pt"}
    if tokenizer.model_max_length < 100000:
        kwargs.update({
            "truncation": True,
            "max_length": tokenizer.model_max_length,
        })
    
    inputs = tokenizer(text, **kwargs)
    full_ids = inputs['input_ids'][0]

    # Create a batch with a row for each length
    batch = full_ids.unsqueeze(0).repeat(len(full_ids), 1)
    labels = torch.full_like(batch, -100)
    
    # Create an identity matrix of size [batch.size(0) x batch.size(0)] and
    # obtain a mask for it
    mask = torch.eye(batch.size(0), dtype=bool)
    
    # Using the created mask, set all values in the labels on the diagonal
    # to the values of the actual batch, so that loss only gets calculated
    # for a single token per row in the batch
    labels[mask] = batch[mask]
    batch[mask] = tokenizer.mask_token_id

    # Drop the first and last rows corresponding to start and end token
    batch = batch[1: -1]
    labels = labels[1: -1]
    
    # Now the batch size is equal to [len(sentence) x len(sentence)]
    # This is too long so I split it to multiple batches of max row count 16
    new_batch = torch.split(batch, batch_size)
    new_labels = torch.split(labels, batch_size)

    # This loop will accumulate the loss over all the
    # 16 length batches
    single_data_point_loss = 0
    with torch.no_grad():
        for b, l in zip(new_batch, new_labels):
            # Send to GPU (if available)
            b = b.to(device)
            l = l.to(device)
            output = model(b, labels=l)
            single_data_point_loss += output.loss.item() * b.shape[0]

    # Calculate BPC
    bpc = single_data_point_loss / (len(text) * np.log(2))
    return bpc

def decoder_full_loss(tokenizer, model, text):
    model.eval()
    device = model.device

    # Check if the model needs to do truncation (some don't have a max length set)
    # (the 100000 is arbitrary but i don't think there's any longer data points in the dataset)
    kwargs = {"return_tensors": "pt"}
    if tokenizer.model_max_length < 100000:
        kwargs.update({
            "truncation": True,
            "max_length": tokenizer.model_max_length,
        })
    inputs = tokenizer(text, **kwargs)

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    
    with torch.no_grad():
        output = model(**inputs, labels=labels)
    
    num_valid_tokens = (labels != -100).sum()
    total_loss = output.loss.item() * num_valid_tokens

    bpc = total_loss / (len(text) * math.log(2))

    return bpc.item()
    
'''
Takes in a list of X BPC values each representing the average at a particular
number of data points being calculated, then determined if they stay within
the prob. range of each other

Parameters
-----------
bpcs        - list of values
prob        - the probability to check

Outputs
------------
outcome     - boolean vale representing if the values are within the prob range of each other
'''
def within_range(bpcs, prob):
    avg_val = sum(bpcs) / len(bpcs)
    spread = max(bpcs) - min(bpcs)

    outcome = (spread / avg_val) <= prob
    return outcome

def lang_len(lm, alpha=0.1, rang=5, encoder=True):
    print(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # load in the model based on if its an encoder or not
    ModelClass = AutoModelForMaskedLM if encoder else AutoModelForCausalLM
    model = ModelClass.from_pretrained(lm).to(device)

    # Open files for writing in data
    dir_path = Path(paths.DATA_DIR) / "lang_lengths" / lm
    dir_path.mkdir(parents=True, exist_ok=True)

    # Get full language name from the current langauge id
    #configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus') # Online
    configs = datasets.get_dataset_config_names(str(paths.GLOTLID)) # Offline

    model_languages = ModelCard.load(lm).data['language']

    # Read in the ISO language dataset
    lang_df = pd.read_csv(f'{paths.DATA_DIR}/language_codes.txt', sep='\t')

    # Convert full language names (referant names) to id names for filtering the datasets later on. Need different ISO formats because of Hugging Face's lack of consistency
    languages = [(i, np.squeeze(lang_df.loc[lang_df['Id'] == i[:3]][['Ref_Name', 'Id', 'Part2b', 'Part2t', 'Part1']].values.tolist())) for i in configs]

    # Determine if languages are present or not in the current model
    languages_to_check = [
        (any(elem in model_languages for elem in a[1]), a)
        for a in languages
    ]
    
    len_dict = {}
    for present, language in languages_to_check:
        print(language)
        # 1. get dataset
        # had to add the try except because some languages seem to be bugged
        try:
            dataset = datasets.load_dataset(str(paths.GLOTLID), language[0]).shuffle()
        except:
            print(f"Can't read in {language}")
            continue
        # 2. create variables for calculating bpc
        n = 0
        bpc, avg_bpc = [], []

        # 3. calculate bpc
        for data_point in dataset['train']:
            # Maintain the number of traversed points
            n += 1
            # 3a) if encoder, go through datapoints one by one
            if encoder:    
                # Retrieve the BPC for the current data point
                res = encoder_full_loss(tokenizer, model, data_point['text'])

            # 3b) if decoder, still go one by one since the avg needs to be
            # calculated for every single data point
            if not encoder:
                res = decoder_full_loss(tokenizer, model, data_point['text'])

            # Add the average and the result to a list for points
            bpc.append(res)
            avg_val = sum(bpc) / len(bpc)
            avg_bpc.append(avg_val)
            print(f'Average BPC at {n} points:', sum(bpc)/len(bpc))

            # If the change within the last rang points is below alpha
            # stop the iteration and append the length required
            if len(bpc) >= rang and within_range(avg_bpc[n-rang:n], alpha):
                len_dict[language[0]] = (n, avg_val)

                # Write JSON to the language presence corresponding file
                with open(dir_path / f"{alpha}_{rang}_{present}_present.json", "a") as f:
                    json.dump({language[0]: {"n": n, "avg_bpc": avg_val}}, f)
                    f.write("\n")
                break

def calc_stats_single(file_path):
    # 1. Read in the data for a single model; present/not present
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
    
    # 2. Get the distribution
    values = np.bincount(population)
    values = values/sum(values)

    # 3. Get the peak of the distribution
    peak = np.argmax(values)

    # 4. Calculate variance
    variance = bpcs.var()
    std_dev = bpcs.std()
    mean = population.mean()
    print(file_path)
    print('peak: ', peak)
    print('variance: ', variance)
    print('std:', std_dev)
    print('mean: ', mean)
    plt.figure()

    
    return population

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

def plot_bpc_vs_n(file):
    bpc_dict = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            curr = json.loads(line)
            for key, value in curr.items():
                bpc_dict[value['n']].append(value['avg_bpc'])

    population = []
    bpcs = []
    with open(file, 'r') as f:
        for line in f:
            curr = json.loads(line)
            for key, value in curr.items():
                population.append(value['n'])
                bpcs.append(value['avg_bpc'])
    population = np.array(population, dtype=int)
    bpcs = np.array(bpcs, dtype=float)
    
    # 2. Get the distribution
    values = np.bincount(population)
    values = values/sum(values)

    # 3. Get the peak of the distribution
    peak = np.argmax(values)

    # 4. Calculate variance
    variance = bpcs.var()
    std_dev = bpcs.std()
    mean = population.mean()

    # Extract only existing lengths and sort them
    lengths = sorted(bpc_dict.keys())
    
    # Calculate the mean only for lengths that exist
    avg_bpc = [np.mean(bpc_dict[l]) for l in lengths]
    
    x = np.array(lengths)
    y = np.array(avg_bpc)

    # window_length must be odd. polyorder is the degree of the polynomial.
    # Tweak these two numbers to increase/decrease the smoothing effect.
    y_smoothed = savgol_filter(y, window_length=11, polyorder=3)

    # Create the figure and the primary axis (ax1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- AXIS 1: The BPC Data (Left side) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of data points (n)')
    ax1.set_ylabel('Average BPC', color=color1)

    # Note: 'x' and 'y' here would be your lengths and smoothed BPC from the previous step
    ax1.plot(x, y_smoothed, color=color1, linewidth=2, label='Avg. BPC')
    ax1.tick_params(axis='y', labelcolor=color1)

    # --- AXIS 2: The Distribution/PDF (Right side) ---
    ax2 = ax1.twinx()  # This is the magic line that creates the right-hand axis

    color2 = 'tab:red'
    ax2.set_ylabel('Population Density (PDF)', color=color2)

    # np.bincount inherently starts at 0, so the x-axis for the PDF is just its length
    x_pdf = np.arange(len(values))

    # Plotting the PDF. A dashed line or bar chart usually looks best for distributions
    ax2.plot(x_pdf, values, color=color2, alpha=0.7, label='Distribution')
    ax2.tick_params(axis='y', labelcolor=color2)

    # --- Clean up and Show ---
    # If your arrays have different max lengths, you might want to set a shared x-limit
    max_x = max(len(y_smoothed), len(values))
    ax1.set_xlim(0, max_x)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title("Average BPC and Population Distribution vs Number of Samples")
    fig.tight_layout() # Prevents the right y-label from getting clipped off the screen
    plt.show()