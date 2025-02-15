import os
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_path, max_sequence_length=350, with_index=False):
        self.file_path = file_path
        self.with_index = with_index

        data_dir = Path(self.file_path).parent
        cached_file_path = f"{data_dir}/dataset-cache.pickle"

        if os.path.exists(cached_file_path):
            cached = torch.load(cached_file_path)
            self.texts = cached["texts"]
            self.labels = cached["labels"]
            self.tokenizer = Tokenizer.from_file(f"{data_dir}/tokenizer.json")
        else:
            df = pd.read_csv('data/OnionOrNot.csv')
            df["text"] = df["text"].apply(lambda text: f"[CLS]{text.lower()}")
            df["text"].to_csv(f"{data_dir}/corpus.txt", index=False, header=False)
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[CLS]", "[PAD]"])
            tokenizer.enable_padding(pad_token="[PAD]", length=max_sequence_length)
            tokenizer.train(["data/corpus.txt"], trainer)
            tokenizer.save(f"{data_dir}/tokenizer.json")
            self.labels = df["label"].tolist()
            self.texts = df["text"].tolist()
            self.tokenizer = tokenizer

            torch.save({
                "texts": self.texts,
                "labels": self.labels,
            }, cached_file_path)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode(text)
        sequence = torch.tensor(encoding.ids)
        attention_mask = torch.tensor(encoding.attention_mask)
        if self.with_index:
            return (sequence, attention_mask), label, idx
        else:
            return (sequence, attention_mask), label

    def __len__(self):
        return len(self.labels)
