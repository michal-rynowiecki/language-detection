import sys

import argparse

from thesis.test_model import test_model

# type in on CL: detect.py
# arguments:
# -lm : langauge model
# -m. : method
# -o. : output path


# Run script to download language model
# Run script to run queries for the model
# Obtain results
# Voila

def main():
    parser = argparse.ArgumentParser(description="Run detection pipeline")
    
    parser.add_argument("-lm", "--language_model", required=True, help="Hugging Face model name")
    parser.add_argument("-m", "--method", required=False, help="Predefined method to run; if empty, all methods will be used")
    parser.add_argument("-o", "--output_path", required=False, help="Output file path")
    parser.add_argument("-l", "--languages", required=False, help="Comma seperated list of languages to test for; if empty, all languages will be tested")

    args = parser.parse_args()

    language_model  = args.language_model
    method          = args.method
    output_path     = args.output_path
    languages       = args.languages

    # Run the src testing methods for the selected model
    languages_input = languages.strip().split(',') # parse list of languages
    test_model(args.language_model, languages=languages_input)

main()