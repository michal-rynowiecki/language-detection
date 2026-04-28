from huggingface_hub import HfApi, ModelCard
from thesis import paths
import sys
from transformers import AutoTokenizer
from datasets import load_dataset 
import json


def api_find():
    api = HfApi()

    models = api.list_models(
        gated=False,
        pipeline_tag="fill-mask",
        num_parameters="min:1B,max:8B",
        sort="downloads"
    )

    # Create a text file that contains already examined models
    # load in the model ids into a list
    path = paths.DATA_DIR / "HF_models.txt"
    path.touch(exist_ok=True)
    with open(path) as f:
        existing_models = [line.strip().split('\t')[0] for line in f]

    for model in models:
        id = model.id
        print(id)
        if (any(substring in id for substring in ['Instruct', 'Distilled', 'Thinking', 'RAG', 'instruct', 'Agent', 'Extract', 'Chat'])
        or model.base_models is not None 
        or id in existing_models):
            continue

        # If so, get model card from HF
        langs = ModelCard.load(id).data['language']
        # Check if list of languages is greater than 2
        if (langs is not None
        and type(langs) is list
        and len(langs) >= 2):
            # Append model to file
            with open(path, 'a') as f:
                f.write(id+'\t'+str(langs)+'\n')

def static_find():
    dataset = load_dataset('cfahlgren1/hub-stats', 'models')["train"]

    sorted_models = dataset.sort('downloadsAllTime', reverse=True)

    for model in dataset:
        id = model["id"]
        if (any(substring in id for substring in ['Instruct', 'Distilled', 'Thinking', 'RAG', 'instruct', 'Agent', 'Extract', 'Chat'])
        or model["baseModels"] is not None):
            continue
        try:
            langs = json.loads(model["cardData"])["language"]
        except:
            continue
        if (langs is not None
            and type(langs) is list
            and len(langs) >= 4):
            print(id, langs)

static_find()