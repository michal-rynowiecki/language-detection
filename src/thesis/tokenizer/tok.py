from transformers import AutoTokenizer

from thesis.method_one.test_lang import tokenizer_test

import pandas as pd

"""
Tokenier based method for detecting the presence of a language 
in a particular language model.

Inputs
--------
@ tok_path: str         - the path to a tokenizer on Hugging Face or local storage
@ languages: list[str]  - list of languages to examine the presence of in the training data

Returns
--------
@ idk
"""
def tokenizer_based(tok_path, languages=['baba']):

    # load in the tokenizer file

    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # Open files for writing in data
    dir_path = Path(paths.DATA_DIR) / "tokenizer_based" / tok_path
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get full language name from the current langauge id
    #configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus') # Online
    configs = datasets.get_dataset_config_names(str(paths.GLOTLID)) # Offline

    tok_languages = .data['language']

    # 1.For language in glotlid:
        # for data_point:
        #   - run the sentence
        #   - average token length
        #   - average Token Fertility (tokens per word)
        #   - number of UNKs
