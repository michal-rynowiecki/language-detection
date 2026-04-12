#python detect.py -lm='google-bert/bert-base-multilingual-cased' -l='French,Danish'
#python determine_lang_length.py -lm='openai-community/gpt2' -r=10 -a=0.1

python determine_lang_length.py -lm google-bert/bert-base-multilingual-cased -r=5 -a=0.1 -en