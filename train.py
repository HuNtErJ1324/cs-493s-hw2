# Training loop for a Transformer model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
import argparse
from utils import Tokenizer, convert_equation_to_str, load_mod_data
from model import GPTConfig, GPT
from torch.utils.data import DataLoader, RandomSampler
import json
from dataclasses import fields


from typing import Optional


def compute_loss(lab, lab_mask, pred_logits):
    loss_per_token = F.cross_entropy(
        pred_logits.permute(0, 2, 1), lab, reduction="none"
    )
    loss = (loss_per_token.view(-1) * lab_mask.view(-1)).sum() / lab_mask.sum()
    return loss


def compute_accuracy(lab, lab_mask, pred_logits):
    pred_tokens = pred_logits.argmax(dim=-1)

    # Step 2: Flatten everything
    pred_flat = pred_tokens.view(-1)
    target_flat = lab.view(-1)
    mask_flat = lab_mask.view(-1)

    # Step 3: Compare predictions to targets where mask == 1
    correct = (pred_flat == target_flat) & (mask_flat.bool())
    accuracy = correct.sum().float() / mask_flat.sum()

    return accuracy


def build_causal_pad_mask(mask: torch.Tensor) -> torch.Tensor:
    B, S = mask.shape # block size
    causal_mask = torch.tril(torch.ones(S, S, device=mask.device))  # (S, S)
    pad_mask = mask.unsqueeze(1).float()  # (B, 1, S)
    combined_mask = causal_mask.unsqueeze(0) * pad_mask.unsqueeze(-2)  # (B, S, S)
    return combined_mask  # 1 for allowed attention, 0 for blocked


def train_one_epoch(model, data, optimizer, config, val_data=None):
    losses = []
    acc = []
    val_losses = []
    val_acc = []
    model.train()
    pbar = tqdm(data, desc="Training")
    for i, bm in enumerate(pbar):
        batch, mask, lab, lab_mask = bm
        batch = batch.to(config.device)
        mask = build_causal_pad_mask(mask).to(config.device)
        lab = lab.to(config.device)
        lab_mask = lab_mask.to(config.device)
        optimizer.zero_grad()
        # Forward
        pred_logits = model(batch, attn_mask=mask)
        # Compute loss only on non padded tokens
        loss = compute_loss(lab, lab_mask, pred_logits)
        losses.append(loss.item())
        acc.append(compute_accuracy(lab, lab_mask, pred_logits).item())

        # Backward pass and param updates
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{losses[-1]:.3f}", acc=f"{acc[-1]:.3f}")

        if config.eval_every and (i + 1) % config.eval_every == 0:
            val_result = validate(model, val_data, config)
            val_losses.append(np.mean(val_result[0]))
            val_acc.append(np.mean(val_result[1]))

    if config.eval_every:
        return losses, acc, val_losses, val_acc

    return losses, acc


def validate(model, val_data, config):
    losses = []
    acc = []
    model.eval()

    # pbar = tqdm(val_data, desc="Validation")
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch, mask, lab, lab_mask in val_data:
            batch = batch.to(device)
            mask = build_causal_pad_mask(mask).to(device)
            lab = lab.to(config.device)
            lab_mask = lab_mask.to(config.device)
            pred_logits = model(batch, attn_mask=mask)
            loss = compute_loss(lab, lab_mask, pred_logits)

            losses.append(loss.item())
            acc.append(compute_accuracy(lab, lab_mask, pred_logits).item())

            # pbar.set_postfix(loss=losses[-1], acc=acc[-1])
    return losses, acc


def train_model(model, train_data, val_data, config, optimizer):
    epoch_loss = []
    epoch_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(config.epochs):
        if config.num_steps > 0:
            epoch_result = train_one_epoch(
                model,
                train_data,
                optimizer,
                config,
                val_data=val_data,
            )
            epoch_loss.append(epoch_result[0])
            epoch_acc.append(epoch_result[1])
            val_loss.append(epoch_result[2])
            val_acc.append(epoch_result[3])
        else:
            epoch_result = train_one_epoch(model, train_data, optimizer)
            epoch_loss.append(epoch_result[0])
            epoch_acc.append(epoch_result[1])
            val_result = validate(model, val_data, config)
            val_loss.append(val_result[0])
            val_acc.append(val_result[1])
        print()
        print(
            "Epoch:",
            epoch,
            " Mean Train Loss: ",
            sum(epoch_loss[-1]) / len(epoch_loss[-1]),
            " Mean Train Accuracy ",
            sum(epoch_acc[-1]) / len(epoch_acc[-1]),
            "Mean Val Loss: ",
            sum(val_loss[-1]) / len(val_loss[-1]),
            "Mean Val Accuracy",
            sum(val_acc[-1]) / len(val_acc[-1]),
        )
    return epoch_loss, epoch_acc, val_loss, val_acc


def test_model(model, test_data, config):
    losses = []
    acc = []
    model.eval()
    pbar = tqdm(test_data, desc="Testing")
    with torch.no_grad():
        for batch, mask, lab, lab_mask in pbar:
            batch = batch.to(config.device)
            mask = build_causal_pad_mask(mask).to(config.device)
            lab = lab.to(config.device)
            lab_mask = lab_mask.to(config.device)
            pred_logits = model(batch, attn_mask=mask)
            loss_per_token = F.cross_entropy(
                pred_logits.flatten(end_dim=1), batch.view(-1), reduction="none"
            )
            loss = compute_loss(lab, lab_mask, pred_logits)
            losses.append(loss.item())
            acc.append(compute_accuracy(lab, lab_mask, pred_logits).item())
            pbar.set_postfix(loss=losses[-1], acc=acc[-1])
    mean_test_loss = sum(losses) / len(losses)
    mean_test_acc = sum(acc) / len(acc)
    print(f"Mean Test Loss: {mean_test_loss:.4f}")
    print(f"Mean Test Accuracy: {mean_test_acc:.4f}")
    return losses, acc


def build_gpt_config(config):
    arg_dict = vars(config)
    field_names = {f.name for f in fields(GPTConfig)}
    filtered_args = {
        k: v for k, v in arg_dict.items() if k in field_names and v is not None
    }
    gpt_confg = GPTConfig(**filtered_args)
    print("Actual GPT config")
    print(str(gpt_confg))
    return gpt_confg


def build_arg_input(**config) -> list[str]:
    arg_list = []
    for key, val in config.items():
        arg_list.extend([f"--{key}", val])
    return arg_list


def parse_args(args: Optional[list]) -> argparse.Namespace:
    # Initial setup of parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=str, choices=["all", "97", "113"], default="all")
    parser.add_argument(
        "--math_op", type=str, choices=["all", "+", "-", "*", "/"], default="all"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")

    # GPT config params
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument(
        "--vocab_size", type=int, default=-1
    )  # -1 if using the vocab size from the dataset

    # Optimizer config
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    parser.add_argument("--exp_name", type=str, default="mod_gpt")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=-1)

    config = parser.parse_args(args)

    # Post validation
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def load_dataset(config: argparse.Namespace) -> None:
    # Load data as tensors
    tokenizer = Tokenizer(block_size=config.block_size)

    # Use for reloading tokenizers for inference
    p_filter = None
    if config.mod != "all":
        p_filter = int(config.mod)
    if config.math_op != "all":
        op_filter = config.math_op

    train_dataset, val_dataset, test_dataset = load_mod_data(
        tokenizer, p_filter=p_filter, op_filter=op_filter
    )
    config.tokenizer = tokenizer.tokenizer
    config.un_tokenizer = tokenizer.un_tokenizer
    # If not predefined we can use vocab size from tokenizer
    if config.vocab_size == -1:
        config.vocab_size = tokenizer.vocab_size
    if config.num_steps > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=RandomSampler(
                train_dataset, True, num_samples=config.num_steps * config.batch_size
            ),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    return train_loader, val_loader, test_loader


def gen_gpt(config: argparse.Namespace):
    gpt_config = build_gpt_config(config)
    gpt = GPT(gpt_config)
    optimizer = gpt.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=config.device,
    )
    gpt.to(config.device)
    print(f"Using device: {next(gpt.parameters()).device}")
    return gpt, optimizer


def __main__(args: list = None) -> None:
    config = parse_args(args)
    print("Config loaded")

    train_loader, val_loader, test_loader = load_dataset(config)
    print("Data loaded")

    # Create the transformer
    gpt, optimizer = gen_gpt(config)
    print("GPT built")
    # Train loop
    print("Train started")
    train_losses, train_acc, val_losses, val_acc = train_model(
        gpt, train_loader, val_loader, config, optimizer
    )
    print("Train done")

    # Testing
    test_losses, test_acc = test_model(gpt, test_loader)

    # Save trained model,configs and losses
    with open(config.exp_name + "_config.json", "w") as f:
        json.dump(vars(config), f, indent=4)

    torch.save(gpt.state_dict(), config.exp_name + ".pth")
    with open(config.exp_name + "_results.json", "w") as f:
        json.dump({
            "train_losses": train_losses, 
            "train_acc": train_acc,
            "val_losses": val_losses, 
            "val_acc": val_acc,
            "test_losses": sum(test_losses) / len(test_losses) if test_losses else None,
            "test_acc": sum(test_acc) / len(test_acc) if test_acc else None,
        }, f)


if __name__ == "__main__":
    __main__()
