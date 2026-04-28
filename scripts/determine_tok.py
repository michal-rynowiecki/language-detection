'''
This script determines the necessary number of data points to use from each language,
after which the average BPC stops changing significantly
'''

import sys

import argparse

import thesis.paths as paths

import pandas as pd
import numpy as np

from thesis.tokenizer.tok import tokenizer_based

def main():
    parser = argparse.ArgumentParser(description="Determine BPC for a given language and token")
    
    parser.add_argument("-p", "--path", required=True, 
        help="path to a file containing a list of tokenizers and their languages")
   
    args = parser.parse_args()

    path  = args.path
    
    tokenizer_based(path)
    
main()