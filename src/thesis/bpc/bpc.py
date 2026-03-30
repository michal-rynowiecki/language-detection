from thesis.bpc.decoder import decode
from thesis.bpc.encoder import encode

from transformers import pipeline

def bpc_based(model_path, languages, encoder=True):

    for lang in languages:
        if encoder:
            encode(model_path, lang)
            break
        else:
            1
        #decode(model, tokenizer, lang)