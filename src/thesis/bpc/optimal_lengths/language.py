# Select a language and a model and its calculated 
# Calculate the BPC for a data point
# Add to a list of totals
# if the change from the last 5 points is not significant, stop

import thesis.paths as paths

import numpy as np
import pandas as pd
import torch

from pathlib import Path

import json

import datasets
from huggingface_hub import ModelCard

from transformers import AutoModelForMaskedLM, AutoTokenizer

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(text, return_tensors='pt')
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
            b.to(device)
            l.to(device)
            output = model(b, labels=l)
            single_data_point_loss += output.loss.item() * b.shape[0]

    # Calculate BPC
    bpc = single_data_point_loss / (len(text) * np.log(2))
    return bpc

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

def lang_len(lm, alpha=0.1, encoder=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = AutoModelForMaskedLM.from_pretrained(lm)
    model.to(device)

    # Open files for writing in data
    dir_path = Path(paths.DATA_DIR) / "lang_lengths" / lm
    dir_path.mkdir(parents=True, exist_ok=True)

    # Get full language name from the current langauge id
    configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus')

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
        if present:
            print(language)
            # 1. get dataset
            dataset = datasets.load_dataset('cis-lmu/glotlid-corpus', language[0]).shuffle()

            # 2. create an empty list for bpc
            n = 0
            bpc = []
            avg_bpc = []
            rang = 5

            # 3. calculate bpc
            # 3a) if encoder, go through datapoints one by one
            if encoder:
                for data_point in dataset['train']:
                    # Maintain the number of traversed points
                    n += 1
                    # Retrieve the BPC for the current data point
                    res = encoder_full_loss(tokenizer, model, data_point['text'])
                    
                    # Add the average and the result to a list for points
                    bpc.append(res)
                    
                    avg_val = sum(bpc) / len(bpc)
                    avg_bpc.append(avg_val)

                    # If the change within the last rang points is below alpha
                    # stop the iteration and append the length required
                    if len(bpc) >= rang:
                        if within_range(avg_bpc[n-rang:n], alpha):
                            len_dict[language[0]] = (n, avg_val)

                            # Write JSON to the language presence corresponding file
                            if present:
                                with open(dir_path / f"{alpha}_present.json", "a") as f:
                                    json.dump({language[0]: {"n": n, "avg_bpc": avg_val}}, f)
                                    f.write("\n")
                            else:
                                with open(dir_path / f"{alpha}_not_present.json", "a") as f:
                                    json.dump({language[0]: {"n": n, "avg_bpc": avg_val}}, f)
                                    f.write("\n")
                            break
                    
                    print(f'Average BPC at {n} points:', sum(bpc)/len(bpc))

            # 3b) if decoder, go through datapoints in batches
            if not encoder:
                n += 1

                # res = ...