from huggingface_hub import HfApi
import os
from chunking.image_chunk import chunk
import nltk
import json
# api = HfApi()

# api.set_access_token('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')

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

if __name__ == "__main__":
    def dummy(prog=None, msg=""):
        if prog:
            print(prog)
        if msg:
            print(msg)
    # res = chunk("./data/index.html", lang="English", callback=dummy)
    res = chunk(filename="./data/test.png", lang="English", callback=dummy)
    print(len(res))
    with open("output.txt", "w") as f:
        f.write(str(res))