import thesis.paths as paths
import datasets

import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer

from huggingface_hub import ModelCard

from pathlib import Path

import json

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
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForMaskedLM.from_pretrained(model_path)

    dataset = datasets.load_dataset('cis-lmu/glotlid-corpus', l[0])
    
    f = open(f'{paths.DATA_DIR}/data_point_lengths/{model_path}/{l[0]}.json', 'w')
    
    f.write(str(language_flag) + '\n')
    i = 0  
    for text in dataset['train']['text']:
        print(i)
        if i == 5:
            break
        
        # Tokenize normally with special tokens
        inputs = tokenizer(text, return_tensors='pt')
        labels = inputs['input_ids'].clone()

        total_loss = 0

        loss_points = []
        for idx, t in enumerate(labels[0]):
            if t.item() in (101, 102):               # Skip the start and stop tokens
                continue
            
            c_inputs = tokenizer(text, return_tensors='pt')
            c_labels = c_inputs['input_ids'].clone()

            c_inputs['input_ids'][0, idx] = tokenizer.mask_token_id
            c_labels[c_inputs['input_ids'] != tokenizer.mask_token_id] = -100
            
            c_str_len = len(tokenizer.decode(labels[0][:idx], skip_special_tokens = True))

            outputs = model(**c_inputs, labels=c_labels)
            total_loss += outputs.loss.item()

            if c_str_len > 0: # Avoid division by 0
                loss_points.append( (c_str_len, total_loss/c_str_len) )
        
        json.dump(loss_points, f)
        f.write('\n')
        i += 1