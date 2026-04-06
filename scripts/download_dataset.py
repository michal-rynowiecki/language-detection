import thesis.paths as paths

from pathlib import Path

from huggingface_hub import snapshot_download

dir_path = Path(paths.DATA_DIR)
dir_path.mkdir(parents=True, exist_ok=True)

folder = snapshot_download(
    "cis-lmu/glotlid-corpus", 
    repo_type="dataset",
    local_dir=f"{dir_path}/glotlid-corpus/",
)