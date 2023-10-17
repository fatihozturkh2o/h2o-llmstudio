# example with OSCAR 2301
# !pip install zstandard

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("oscar-corpus/OSCAR-2301",
                        use_auth_token="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", # required 
                        language="tr", 
                        streaming=True, # optional
                        split="train") # optional


i = 0
n_samples = 10_000
samples = []
for d in dataset:
    samples.append(
        [d["text"]]
    )
    i+=1
    if i % 1000 == 0:
        print(f"Step: {i}")
    if i==n_samples:
        break

df = pd.DataFrame(samples, columns=["prompt"])
df["answer"] = ""
df.to_csv(f"oscar_tr_ds_{n_samples}.csv")
