import ipdb
import pandas as pd
from os import listdir
from os.path import isfile, join
from functools import reduce

train_files = ["train/" + f for f in listdir("train")]

train_dfs = []

for f in train_files:
    train_df = pd.read_csv(f, sep='\t', header=None,
                           names=["id", "sentiment", "text"],
                           usecols=["id", "sentiment", "text"])
    train_dfs.append(train_df)

overall_df = pd.concat(train_dfs)
overall_df.to_csv(path_or_buf="train/OVERALL.tsv",
                  sep=" ", header=None, index=False)
