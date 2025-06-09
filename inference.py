# Training loop for a Transformer model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
import argparse
from argparse import Namespace
from utils import Tokenizer, convert_equation_to_str, load_mod_data
from model import GPTConfig, GPT
from train import build_gpt_config
from torch.utils.data import DataLoader
import json


def __main__():
    # Initial setup of parameters
    parser = argparse.ArgumentParser(
        description=(
            """
            Executes model inference for a model and dataset specified with seed_file and mod_dataset. Output is written to out_file. 
            
            Example usage: python inference.py --use_mod_dataset --seed_file /path/to/grok.pth --out_file /path/to/out_file.txt
            """
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--use_mod_dataset",
        action="store_true",
        help="whether to use the existing modulo dataset in the submitted code",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=-1,
        help="maximum sequence length. The default value uses maximum within dataset.",
    )  # then uses box size
    parser.add_argument(
        "--num_samples", type=int, default=16, help="the number of samples to infer"
    )
    parser.add_argument(
        "--seed_file", type=str, default=None, help="full path to pretrained model file"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="full path to output file where inference will be stored",
    )
    parser.add_argument(
        "--print_samples",
        type=bool,
        default=True,
        help="whether to also print samples in addition to storing in out_file",
    )

    parser.add_argument("--exp_name", type=str, default="mod_gpt", help="name of the experiment")
    infer_configs = parser.parse_args()

    with open(infer_configs.exp_name + "_config.json", "r") as f:
        config = Namespace(**json.load(f))
    print("Config loaded")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if infer_configs.seq_len == -1:
        infer_configs.seq_len = config.block_size
    # Load data as tensors
    tokenizer = Tokenizer(
        block_size=config.block_size,
        tokenizer=config.tokenizer,
        un_tokenizer=config.un_tokenizer,
    )

    samples = torch.zeros(
        infer_configs.num_samples, infer_configs.seq_len, dtype=torch.long
    ).to(device)
    if infer_configs.seed_file:
        # TODO

        # should provide co<mpatability for more advanced seeding of sampling
        pass
    else:
        init_len = 1
        for i in range(infer_configs.num_samples):
            samples[i, 0] = tokenizer.tokenizer[
                np.random.choice(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            ]
    # Create the transformer
    gpt_config = build_gpt_config(config)
    gpt = GPT(gpt_config)
    gpt.to(device)
    gpt.load_state_dict(torch.load(config.exp_name + ".pth", map_location=device))
    gpt.eval()
    init_len = 1
    if infer_configs.seq_len == -1:
        infer_configs.seq_len = config.block_size
    with torch.no_grad():
        for i in range(init_len, infer_configs.seq_len - 1):
            samples[:, i] = gpt(samples).argmax(dim=2)[:, min(i, config.block_size - 1)]
    samples = samples.cpu().numpy().astype(int).astype(str)
    samples_as_text = tokenizer.untokenize(samples)
    if infer_configs.print_samples:
        for i in range(infer_configs.num_samples - 1):
            print(samples_as_text[i])


if __name__ == "__main__":
    __main__()
