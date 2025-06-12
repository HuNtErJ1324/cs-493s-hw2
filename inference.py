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


def __main__(args: list = None):
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
        "--seq_len",
        type=int,
        default=-1,
        help="maximum sequence length. The default value uses maximum within dataset.",
    )  # then uses box size
    parser.add_argument(
        "--num_samples", type=int, default=16, help="the number of samples to infer"
    )
    parser.add_argument(
        "--data_path", type=str, default="part2/data_test.csv", help="path to data for seeding"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="full path to output file where inference will be stored, if none no file is created",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to use argmax or random sampling for inference",
    )
    parser.add_argument(
        "--print_samples",
        action="store_false",
        help="whether to also print samples in addition to storing in out_file",
    )
    parser.add_argument("--model_name", type=str, default="mod_gpt", help="location of the model that performs the inference and its configs")
    infer_configs = parser.parse_args(args)

    with open(infer_configs.model_name + "_config.json", "r") as f:
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
    )
    if infer_configs.data_path: # read seed data from file, expects same format as the mod files
        if config.mod != "all":
            p_filter = int(config.mod)
        if config.math_op != "all":
            op_filter = config.math_op
        test = pd.read_csv(infer_configs.data_path)
        if p_filter:
            print(f"Filtering dataset for p={p_filter}")
            test = test[test["p"] == p_filter].reset_index(drop=True)

        if op_filter:
            print(f"Filtering dataset for op={op_filter}")
            test = test[test["o"] == op_filter].reset_index(drop=True)
        convert_equation_to_str(test)
        test_questions, _ = tokenizer.tokenize(test["q_str"])
        samples = torch.from_numpy(test_questions[:infer_configs.num_samples])
        samples_infer = np.zeros(infer_configs.num_samples) + 2
    else: # Random sampling, seeds the sequence with one random char from the vocab and then produces a sequence
        init_len = 1
        samples[:, 0] = torch.Tensor(np.random.choice(tokenizer.vocab_size,size=infer_configs.num_samples))
        samples_infer = np.zeros(infer_configs.num_samples) + infer_configs.seq_len-init_len
    # Create the transformer
    gpt_config = build_gpt_config(config)
    gpt = GPT(gpt_config)
    gpt.to(device)
    gpt.load_state_dict(torch.load(infer_configs.model_name + ".pth", map_location=device))
    gpt.eval()
    samples = samples.to(device)
    init_len = 1
    if infer_configs.seq_len == -1:
        infer_configs.seq_len = config.block_size
    with torch.no_grad():
        for i in range(init_len, infer_configs.seq_len - 1):
            logits = gpt(samples) 
            logits = logits[:, min(i, config.block_size - 1), :]
            probs = torch.softmax(logits, dim=-1) 
            for j in range(infer_configs.num_samples):
                p = probs[j].cpu().numpy()
                if samples[j,i] == tokenizer.tokenizer['<pad>'] and samples_infer[j] > 0:
                    samples_infer[j] -= 1
                    if infer_configs.deterministic:
                        samples[j,i] = p.argmax()
                    else:
                        samples[j, i] = np.random.choice(a=tokenizer.vocab_size, p=p)
    samples = samples.cpu().numpy().astype(int).astype(str)
    samples_as_text = tokenizer.untokenize(samples)
    if infer_configs.print_samples:
        for i in range(infer_configs.num_samples - 1):
            print(samples_as_text[i])
            
    if infer_configs.out_file:
        with open(infer_configs.out_file, "w") as file:
            file.writelines(samples_as_text)



if __name__ == "__main__":
    __main__()
