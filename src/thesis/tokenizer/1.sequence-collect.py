import sys
from transformers import AutoTokenizer
from datasets import load_dataset 
import json

#dataset = load_dataset('cfahlgren1/hub-stats', 'models')
dataset = load_dataset('cfahlgren1/hub-stats', 'models', revision='eaabe50c4606f6db3313bfb9d823b82b2c56bbcd')

# Could also use 'downloads' for last 30 days
sorted_models = dataset.sort('downloadsAllTime', reverse=True)

paths = {}

tokenizers = []
pathsfile = open('1.sequences.txt', 'w')
for entry in sorted_models['train']:
    langs = ""
    if len(tokenizers) >= 500:
        break
    lm = entry['id']
    try:
        langs = json.loads(entry['cardData'])['language']
    except:
        print('no langs listed')
    print(langs)
    try:
        # Some models also have gotten a backend tokenizer later (only on the
        # fast one), for example the original BERT, NFC used to be in the code
        # only, but now it is also in the fast tokenizer as a regular
        # pre_tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(lm, trust_remote_code=True)
        pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
        print(tokenizer.backend_tokenizer.pre_tokenizer)
        pre_tokenizer_types = [str(pre_tokenizer).split('(')[0]]
        if pre_tokenizer_types[0] == 'Sequence':
            pre_tokenizer_types = [str(step).split('(')[0] for step in pre_tokenizer]
        if pre_tokenizer_types == ['None']:
            pre_tokenizer_types = []
        main_tokenizer = str(tokenizer.backend_tokenizer.model).split('(')[0]
        path = pre_tokenizer_types + [main_tokenizer]
        print(path)
        pathsfile.write(lm + '\t' + str(langs) + '\t' + str(path) + '\n')
        tokenizers.append(tokenizer)


    except ImportError as v : # not the right packages installed, probably non text
        print('ImportError', lm)
    except KeyError as v :
        print('KeyError', lm) # Non-text models have no tokenizer mapping
    except ValueError as v :
        print('ValueError', lm) # Unknown model
    except TypeError as v :
        print('ValueError', lm) # Non-text models might have no vocab file
    except AttributeError as v:
        print('AttributeError', lm) # non standard tokenizer
    except OSError as v:
        print('OSError', lm) # no access
    except ZeroDivisionError as v:
        print('0divisionError', lm) # there is some error in the tokenizer
    except BaseException as v:
        print('pyo3_runtime.PanicException (or something else)', lm) # pyo3_runtime.PanicException: byte index 100 is not a char boundary; it is inside

        

# Now compare all tokenizers to all tokenizers to find duplicates.
pathsfile.close()
counter = 0
for tokenizer1_idx, tokenizer1 in enumerate(tokenizers):
    matched = False
    for tokenizer2_idx, tokenizer2 in enumerate(tokenizers):
        if tokenizer1_idx != tokenizer2_idx and tokenizer1  == tokenizer2:
            matched =True
            break
    if matched:
        counter += 1
print(counter, len(tokenizers))
