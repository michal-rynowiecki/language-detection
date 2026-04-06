#!/bin/bash

while read -r arg1; do
    python determine_string_length.py "-lm=prajjwal1/bert-tiny" "-l=$arg1"
done < /Users/michal/Projects/Thesis/data/glotlid_langs.txt
