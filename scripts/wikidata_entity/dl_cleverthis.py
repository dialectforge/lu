#!/usr/bin/env python3
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="CleverThis/wikidata-truthy", repo_type="dataset",
                      allow_patterns=["data/*.parquet"],
                      local_dir="/var/lib/lu/cleverthis")
print("DONE_DL", p, flush=True)
