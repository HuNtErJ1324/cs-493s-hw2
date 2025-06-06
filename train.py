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
from torch.utils.data import DataLoader
import json

def train_one_epoch(model,data,optimizer):
    losses = []
    model.train()
    pbar = tqdm(data, desc="Training")
    device = next(model.parameters()).device
    for batch,mask in pbar:
        batch = batch.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        # Forward
        pred_logits = model(batch,attn_mask=mask)
        # Compute loss only on non padded tokens
        loss_per_token = F.cross_entropy(pred_logits.flatten(end_dim=1),batch.view(-1),reduction='none')
        loss = (loss_per_token * mask.view(-1)).sum() / mask.sum()
        losses.append(loss.item())
        # Backward pass and param updates
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())
    return losses

def validate(model,val_data):
    losses = []
    model.eval()
    
    pbar = tqdm(val_data, desc="Validation")
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch,mask in pbar:
            batch = batch.to(device)
            mask = mask.to(device)
            pred_logits = model(batch,attn_mask=mask)
            loss_per_token = F.cross_entropy(pred_logits.flatten(end_dim=1),batch.view(-1),reduction='none')
            loss = (loss_per_token * mask.view(-1)).sum() / mask.sum()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
    return losses

def train_model(model,train_data,val_data,epochs,optimizer):
    epoch_loss = []
    val_loss = []
    for epoch in range(epochs):
        epoch_loss.append(train_one_epoch(model,train_data,optimizer))
        val_loss.append(validate(model,val_data))
        print("Epoch:", epoch, " Mean Train Loss: ", sum(epoch_loss[-1])/len(epoch_loss[-1]), "Mean Val Loss: ", sum(val_loss[-1])/len(val_loss[-1]))
    return epoch_loss,val_loss

def test_model(model, test_data):
    losses = []
    model.eval()
    device = next(model.parameters()).device
    pbar = tqdm(test_data, desc="Testing")
    with torch.no_grad():
        for batch, mask in pbar:
            batch = batch.to(device)
            mask = mask.to(device)
            pred_logits = model(batch, attn_mask=mask)
            loss_per_token = F.cross_entropy(pred_logits.flatten(end_dim=1),batch.view(-1),reduction='none')
            loss = (loss_per_token * mask.view(-1)).sum() / mask.sum()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
    mean_test_loss = sum(losses) / len(losses)
    print(f"Mean Test Loss: {mean_test_loss:.4f}")
    return losses
    

def build_gpt_config(config):
    gpt_config = GPTConfig()
    return gpt_config

def __main__():
    # Initial setup of parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_dataset', type=bool,default=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--criterion',type=str,default='CrossEntropyLoss')
    
    # GPT config params
    parser.add_argument('--n_layer',type=int,default=12)
    parser.add_argument('--n_head',type=int,default=12)
    parser.add_argument('--n_embd',type=int,default=768)
    parser.add_argument('--dropout',type=float,default=0.0)
    parser.add_argument('--bias',type=bool,default=True)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--block_size',type=int,default=16)
    parser.add_argument('--vocab_size',type=int,default=-1) # -1 if using the vocab size from the dataset
    
    # Optimizer config
    parser.add_argument('--learning_rate',type=float,default=6e-4)
    parser.add_argument('--weight_decay',type=float,default=1e-1)
    parser.add_argument('--beta1',type=float,default=0.9) 
    parser.add_argument('--beta2',type=float,default=0.95)
    
    
    parser.add_argument('--exp_name',type=str,default="mod_gpt")

    config = parser.parse_args()
    print("Config loaded")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Load data as tensors
    
    tokenizer = Tokenizer(block_size=config.block_size)
    # Use for reloading tokenizers for inference
    if config.mod_dataset:
      train_dataset,val_dataset,test_dataset = load_mod_data(tokenizer)
    config.tokenizer = tokenizer.tokenizer
    config.un_tokenizer = tokenizer.un_tokenizer
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    # If not predefined we can use vocab size from tokenizer
    if config.vocab_size == -1:
      config.vocab_size = tokenizer.vocab_size
    print("Data loaded")
    
    # Create the transformer  
    gpt_config = build_gpt_config(config)
    gpt = GPT(gpt_config)
    optimizer = gpt.configure_optimizers(config.weight_decay, config.learning_rate,betas=(config.beta1,config.beta2),device_type=device)
    gpt.to(device)
    print(next(gpt.parameters()).device)
    print("GPT built")
    # Train loop
    print("Train started")
    train_losses,val_losses = train_model(gpt,train_loader,val_loader,config.epochs,optimizer)
    print("Train done")

    # Testing
    test_model(gpt,test_loader)
    
    # Save trained model,configs and losses
    with open(config.exp_name + "_config.json", "w") as f:
        json.dump(vars(config), f, indent=4)

    
    torch.save(gpt.state_dict(), config.exp_name+".pth")
    
    with open(config.exp_name+"_losses.json", "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f)

    




if __name__ == "__main__":
    __main__()
