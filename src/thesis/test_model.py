import pandas as pd

from thesis.method_one.tok import tokenizer_based
from thesis.method_two.bpc import bpc_based
import thesis.paths as paths

def test_model(model_path, method='all', languages='test'):
    # Read in the ISO language dataset
    lang_df = pd.read_csv(f'{paths.DATA_DIR}/language_codes.txt', sep='\t')
    
    # Convert full language names (referant names) to id names for filtering the datasets later on. Need different ISO formats because of Hugging Face's lack of consistency
    languages_input = [np.squeeze(lang_df.loc[lang_df['Ref_Name'] == l][['Id', 'Part2b', 'Part2t', 'Part1']].values).tolist() for l in languages]

    #Obtain the model and tokenizer

    #tokenizer_based(model_path)
    bpc_based(model_path, languages_input)
