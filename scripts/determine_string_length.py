import sys

import argparse

import thesis.paths as paths

import pandas as pd
import numpy as np

from thesis.bpc.optimal_lengths.data_point import str_len

def main():
    parser = argparse.ArgumentParser(description="Determine BPC for a given language and token")
    
    parser.add_argument("-lm", "--language_model", required=True, help="Hugging Face model name")
    parser.add_argument("-l", "--language", required=True, help="Languages to test for")

    args = parser.parse_args()

    language_model  = args.language_model
    language        = args.language
    print(language)
    # Read in the ISO language dataset
    lang_df = pd.read_csv(f'{paths.DATA_DIR}/language_codes.txt', sep='\t')
    
    # Convert full language names (referant names) to id names for filtering the datasets later on. Need different ISO formats because of Hugging Face's lack of consistency
    lang = np.squeeze(lang_df.loc[lang_df['Id'] == language[:3]][['Id', 'Part2b', 'Part2t', 'Part1']].values.tolist())

    print(lang)
    # Run the src testing methods for the selected model
    str_len(args.language_model, lang=language)

main()

