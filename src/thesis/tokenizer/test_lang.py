from datasets import get_dataset_config_names, load_dataset

from thesis import paths
"""
Takes in a text data sample and a tokenizer and calculates the average and median
token length after tokenization

Parameters
-----------
@ sample: str           - text data sample to be tokenized
@ tokenizer: tokenizer  - tokenizer to process the data sample

Outputs
-----------
@ avg: float            - average length of the obtained tokens
@ med: float            - median length of the obtained tokens
"""
def check_sample(sample: str, tokenizer: AutoTokenizer) -> tuple[float, float]:
    tokenized = tokenizer.tokenize(sample)
    lengts = [len(token) for token in tokenized]
    return avg(lengths), median(lenghts)

"""
Examines the presence of the provided langauges by using the selected dataset for probes
and selected tokenizer for obtaining subtokens

Parameters
-----------
@ tokenizer: AutoTokenizer  - 
"""
def tokenizer_test(tokenizer: AutoTokenizer, language: str, dataset='glotlid-corpus'):
    # 1. load in dataset from data directory
    # 2. obtain list of languages in the dataset
    dataset_path = f'{paths.DATA_DIR}/train_datasets/{dataset}'
    config = get_dataset_config_names(dataset_path)
    print(config[0])

    # 3. create a dictionary for storing averages and medians
    dict = {}
    # 4. for langauge:
    language_data = load_dataset(dataset_path)
    print(language_data, config[0])
    #       for every sample:
    #           avg, median = check_sample(sample, tokenizer)
    #           dict[language].append( (avg, median) )
    # 5. Calculate averages of averages for langauges
    # return avg, med