import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from model import GPTConfig, GPT
from typing import Optional


class Tokenizer:
    def __init__(self, block_size, tokenizer=None, un_tokenizer=None):
        self.tokenizer = tokenizer
        self.un_tokenizer = un_tokenizer
        self.block_size = block_size
        if tokenizer:
            self.vocab_size = len(self.tokenizer.keys())
        else:
            self.vocab_size = None

    def build_tokenization(self, train_sentences):
        # character level tokenization
        vocab = set("".join(train_sentences))
        self.tokenizer = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.tokenizer["<pad>"] = 0
        vocab.add("<pad>")
        self.un_tokenizer = {v: k for k, v in self.tokenizer.items()}
        self.vocab_size = len(vocab)

    def tokenize(self, sentences):
        encoded = np.full(
            (len(sentences), self.block_size), self.tokenizer["<pad>"], dtype=int
        )
        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence[: self.block_size]):
                encoded[i, j] = self.tokenizer.get(char, self.tokenizer["<pad>"])
        attention_mask = ~(encoded == self.tokenizer["<pad>"])
        return encoded, attention_mask

    def untokenize(self, encoded):
        sentences = []
        for i in range(len(encoded)):
            sentence = ""
            for j in range(len(encoded[i])):
                if encoded[i][j] != self.tokenizer["<pad>"]:
                    sentence += self.un_tokenizer[encoded[i][j]]
            sentences.append(sentence)
        return sentences


def load_mod_data(tokenizer, data_path="part2/", p_filter: Optional[int] = None, op_filter: Optional[str] = None):
    # Read files
    train = pd.read_csv(data_path + "data_train.csv")
    val = pd.read_csv(data_path + "data_val.csv")
    test = pd.read_csv(data_path + "data_test.csv")

    if p_filter:
        print(f"Filtering dataset for p={p_filter}")
        train = train[train["p"] == p_filter].reset_index(drop=True)
        val = val[val["p"] == p_filter].reset_index(drop=True)
        test = test[test["p"] == p_filter].reset_index(drop=True)

    if op_filter:
        print(f"Filtering dataset for op={op_filter}")
        train = train[train["o"] == op_filter].reset_index(drop=True)
        val = val[val["o"] == op_filter].reset_index(drop=True)
        test = test[test["o"] == op_filter].reset_index(drop=True)
    
    # Create strs from equations
    convert_equation_to_str(train)
    convert_equation_to_str(val)
    convert_equation_to_str(test)

    # Build tokenizer
    tokenizer.build_tokenization(train["as_str"])

    # Tokenize
    train_tokens, train_mask = tokenizer.tokenize(train["q_str"])
    val_tokens, val_mask = tokenizer.tokenize(val["q_str"])
    test_tokens, test_mask = tokenizer.tokenize(test["q_str"])

    train_label_tokens, train_label_mask = tokenizer.tokenize(train["a_str"])
    val_label_tokens, val_label_mask = tokenizer.tokenize(val["a_str"])
    test_label_tokens, test_label_mask = tokenizer.tokenize(test["a_str"])

    train = torch.from_numpy(train_tokens)
    train_mask = torch.from_numpy(train_mask)
    train_labels = torch.from_numpy(train_label_tokens)
    train_label_masks = torch.from_numpy(train_label_mask)

    val = torch.from_numpy(val_tokens)
    val_mask = torch.from_numpy(val_mask)
    val_labels = torch.from_numpy(val_label_tokens)
    val_label_masks = torch.from_numpy(val_label_mask)

    test = torch.from_numpy(test_tokens)
    test_mask = torch.from_numpy(test_mask)
    test_labels = torch.from_numpy(test_label_tokens)
    test_label_masks = torch.from_numpy(test_label_mask)

    train_dataset = TensorDataset(train, train_mask, train_labels, train_label_masks)
    val_dataset = TensorDataset(val, val_mask, val_labels, val_label_masks)
    test_dataset = TensorDataset(test, test_mask, test_labels, test_label_masks)

    print(f"Train dataset size: {len(train)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def _gen_question(df):
    return df[["a", "o", "b"]].astype(str).agg(" ".join, axis=1) + " = "


def convert_equation_to_str(df):
    df["as_str"] = _gen_question(df) + df["c"].astype(str)
    df["q_str"] = _gen_question(df)
    df["a_str"] = df["c"].astype(str)
