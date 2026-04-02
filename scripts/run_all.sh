#!/bin/bash

while read -r arg1; do
    python determine_string_length.py "-lm=google-bert/bert-base-multilingual-cased" "-l=$arg1"
done < /Users/michal/Projects/Thesis/data/glotlid_langs.txt
