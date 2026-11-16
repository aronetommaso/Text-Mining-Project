import pandas as pd
import os
import numpy as np
from datasets import load_dataset

ds = load_dataset("QCRI/HumAID-all", verification_mode="no_checks")

ds_train = ds["train"].to_pandas()

print(ds_train.head())

print(ds_train["class_label"].value_counts())

print(ds_train[ds_train["class_label"] == "requests_or_urgent_needs"].iloc[0]["tweet_text"]) # Word "Help"

