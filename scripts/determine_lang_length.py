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
from thesis.bpc.optimal_lengths.plot import calc_stats_single, plot_bpc_vs_n, plot_true_vs_false

def main():
    parser = argparse.ArgumentParser(description="Determine BPC for a given language and token")
    
    parser.add_argument("-lm", "--language_model", required=True, 
        help="Hugging Face model name")
    
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
        help="The spread that r consecutive values must be in")

    parser.add_argument("-r", "--rang", type=int, default=5,
        help="Number of consecutive values")

    parser.add_argument("-en", "--encoder", action="store_true",
        help="Is the model an encoder?")
   
    args = parser.parse_args()

    language_model  = args.language_model
    alpha = args.alpha
    rang = args.rang
    encoder = args.encoder
    
    parser.set_defaults(encoder=False)

    # Run the src method for determining optimal number of data points
    lang_len(args.language_model, alpha, rang, encoder)

    #calc_stats_single('/Users/michal/Projects/Thesis/data/lang_lengths/bert-base-multilingual-cased')
    #plot_bpc_vs_n('/Users/michal/Projects/Thesis/data/lang_lengths/bert-base-multilingual-cased/0.01_5_True_present.json')
    €plot_true_vs_false('/Users/michal/Projects/Thesis/data/lang_lengths/bert-base-multilingual-cased/0.001_20_True_present.json', '/Users/michal/Projects/Thesis/data/lang_lengths/bert-base-multilingual-cased/0.001_20_False_present.json')

main()