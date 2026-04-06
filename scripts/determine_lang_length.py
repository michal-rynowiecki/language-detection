'''
This script determines the necessary number of data points to use from each language,
after which the average BPC stops changing significantly
'''

import sys

import argparse

import thesis.paths as paths

import pandas as pd
import numpy as np

from thesis.bpc.optimal_lengths.language import lang_len

def main():
    parser = argparse.ArgumentParser(description="Determine BPC for a given language and token")
    
    parser.add_argument("-lm", "--language_model", required=True, help="Hugging Face model name")

    args = parser.parse_args()

    language_model  = args.language_model

    # Run the src method for determining optimal number of data points
    lang_len(args.language_model, encoder=False)

main()
