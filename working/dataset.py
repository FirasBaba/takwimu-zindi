import os

import pandas as pd
from torch.utils.data import Dataset

class YorubaDatasetSpecial(Dataset):
    def __init__(self, tokenizer, csv_path, max_len_french=100, max_len_target=100, task="train"):
        self.csv_path = csv_path
        self.source_column = "sent1"
        self.target_column = "sent2"
        self.data = pd.read_csv(self.csv_path) #.sample(100).reset_index()
        self.max_len_french = max_len_french
        self.max_len_target = max_len_target
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_, target = (
                self.data.loc[index, self.source_column],
                self.data.loc[index, self.target_column],
            )
            
        target_language = self.data.loc[index, "target_language"]

        input_ = f"translate French to {target_language}: " + str(input_) + " </s>"
        target = f"translate French to {target_language}: " + str(target) + " </s>"

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_],
            truncation=True,
            max_length=self.max_len_french,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target],
            truncation=True,
            max_length=self.max_len_target,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs[
            "attention_mask"
        ].squeeze()  # might need to squeeze
        target_mask = tokenized_targets[
            "attention_mask"
        ].squeeze()  # might need to squeeze

        return {
            "en_sentence" : target,
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }