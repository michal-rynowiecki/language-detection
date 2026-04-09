import thesis.paths as paths
import datasets

import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer,  BertForMaskedLM

from huggingface_hub import ModelCard

from pathlib import Path

import json

'''
Takes in a model for Masked LM and returns the loss obtained by
masking each token seperately in a batch
'''
def encoder_full_loss(model, input):
    inputs = tokenizer(text, return_tensors='pt')
    full_ids = inputs['input_ids'][0]

    # Create a batch with a row for each length
    batch = full_ids.unsqueeze(0).repeat(len(full_ids), 1)
    labels = torch.full_like(batch, -100)
    
    mask = torch.eye(batch.size(0), dtype=bool)
    labels[mask] = batch[mask]

    batch = batch[1: -1]
    labels = labels[1: -1]
    print(batch.shape)
    print('Batch:', batch)
    print(labels.shape)
    print('Labels: ', labels)

    output = model(batch, labels=labels)

    print(loss)

def str_len(model_path, lang):
    Path(f'{paths.DATA_DIR}/data_point_lengths/{model_path}').mkdir(parents=True, exist_ok=True)

    # Get full language name from the current langauge id
    configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus')

    model_languages = ModelCard.load(model_path).data['language']

    # Create a tuple: (glotlid language ID, ISO language names)
    l = next(((c, lang) for c in configs if c[:3] in lang), None)

    model_languages = ModelCard.load(model_path).data['language']
    
    # Check if current language is present in the model training data
    if not isinstance(model_languages, list):
        model_languages = [model_languages]

    language_flag = any(model_language in l[1] for model_language in model_languages)
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model =  BertForMaskedLM.from_pretrained(model_path)

    dataset = datasets.load_dataset('cis-lmu/glotlid-corpus', l[0])
    
    f = open(f'{paths.DATA_DIR}/data_point_lengths/{model_path}/{l[0]}.json', 'w')
    
    f.write(str(language_flag) + '\n')
    i = 0  
    # First tokenize the text
    for text in dataset['train']['text']:
        # Tokenize normally with special tokens
        inputs = tokenizer(text, return_tensors='pt')
        #labels = inputs['input_ids'].clone()
        full_ids = inputs['input_ids'][0]

        print('Inputs: ', inputs)
        print('FULL IDS: ', full_ids)

        # Create a batch with a row for each length
        batch = full_ids.unsqueeze(0).repeat(len(full_ids), 1)
        labels = torch.full_like(batch, -100)
        
        mask = torch.eye(batch.size(0), dtype=bool)
        labels[mask] = batch[mask]

        batch = batch[1: -1]
        labels = labels[1: -1]
        print(batch.shape)
        print('Batch:', batch)
        print(labels.shape)
        print('Labels: ', labels)

        output = model(batch, labels=labels)
        print(output)
        # Each batch should contain, on every row, a single masked token
        break
        
