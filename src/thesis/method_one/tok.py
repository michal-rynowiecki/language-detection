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

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    print(lala)
    language_df = pd.read_csv(f'{paths.DATA_DIR}/language_codes.txt', sep='\t')
    
    '''
    # Run X number of sentences through the tokenizer
    if languages=='all':
        # Go through all languages
        1
    else:
        for language in languages:
            tokenizer_test(tokenizer, language)
            # Go through the language

    # Get the average token length (compared to gibbresih (?)) and check if it is below the treshold

    # Return 0 or 1 for a particular language

    return 1
    '''