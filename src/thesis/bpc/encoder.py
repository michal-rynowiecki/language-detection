# TODO:
# 1. Check what length of a sentence starts giving the same amount of perplexity

import thesis.paths as paths
import datasets

import torch

import numpy as np

from transformers import AutoModelForMaskedLM, AutoTokenizer

def encode(model_path, lang):
    # Get full language name from the current langauge id
    configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus')
    print(lang)
    for i in configs:
        print(i)
    l = next(((c, lang) for c in configs if c[:3] in lang), None)

    print(l)
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get the model
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    # Load in the dataset
    dataset = datasets.load_dataset('cis-lmu/glotlid-corpus', l[0])

    # Split into batches
    dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size = 16)
    
    text = dataset['train']['text'][0]

    # Tokenize normally with special tokens
    inputs = tokenizer(text, return_tensors='pt')
    labels = inputs['input_ids'].clone()
    print(tokenizer.mask_token_id)
    
    total_loss = 0

    for idx, t in enumerate(labels[0]):
        if t.item() in (101, 102):
            continue
        c_inputs = tokenizer(text, return_tensors='pt')
        c_labels = c_inputs['input_ids'].clone()

        c_inputs['input_ids'][0, idx] = tokenizer.mask_token_id
        c_labels[c_inputs['input_ids'] != tokenizer.mask_token_id] = -100
        
        c_str_len = len(tokenizer.decode(labels[0][:idx], skip_special_tokens = True))

        outputs = model(**c_inputs, labels=c_labels)
        
        total_loss += outputs.loss.item()

        if c_str_len > 0:
            print(f'BPC at index {idx} and str length {c_str_len}:', total_loss / (c_str_len * np.log(2)))

    bpc = total_loss / (c_str_len * np.log(2))
    print('Text length: ', len(text))
    print(bpc)