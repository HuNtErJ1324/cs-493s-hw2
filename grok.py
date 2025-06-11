import train
import json
import torch
import numpy as np
import argparse

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



parser = argparse.ArgumentParser()
parser.add_argument("--run_opt", type=int, choices=[1, 2, 3], default=1)
cfg = parser.parse_args()

grok_dir = "grokking"

# Setup 
args = {
    'mod': '97',
    "math_op": "/",
    "epochs": "1",
    "n_layer": "2",
    "n_head": "4",
    "n_embd": "128",
    "block_size": "32",  # [x, y, =] or [x, y, z]
    "dropout": "0.0",
    "bias": "True",
    "learning_rate": "1e-3",
    "weight_decay": "1e-1",
    "batch_size": "512",
    'exp_name': "grok",
    "eval_every": "1000",
    "num_steps": "1_000_000",
}

if cfg.run_opt == 1:
    args["weight_decay"] = "1"
elif cfg.run_opt == 2:
    args["n_layer"] = "1"
    args["n_head"] = "2"
    args["n_embd"] = "64"
elif cfg.run_opt == 3:
    args["weight_decay"] = "1"
    args["n_layer"] = "1"
    args["n_head"] = "2"
    args["n_embd"] = "64"

config = train.parse_args(train.build_arg_input(**args))
print("Config loaded. Using config")

train_loader, val_loader, test_loader = train.load_dataset(config)
print("Data loaded")


# Create the transformer
gpt, optimizer = train.gen_gpt(config)
print("GPT built")
# Train loop
print("Train started")
train_losses, train_acc, val_losses, val_acc = train.train_model(
    gpt, train_loader, val_loader, config, optimizer
)
print("Train done")

# Testing
train.test_model(gpt, test_loader, config)

# Save trained model,configs and losses
with open(f"{grok_dir}/{config.exp_name}_config_opt{cfg.run_opt}.json", "w") as f:
    json.dump(vars(config), f, indent=4)

torch.save(gpt.state_dict(), f"{grok_dir}/{config.exp_name}_opt{cfg.run_opt}.pth")

with open(f"{grok_dir}/{config.exp_name}_losses_opt{cfg.run_opt}.json", "w") as f:
    json.dump({"train_losses": train_losses[0], "val_losses": val_losses}, f)


np.save(f"{grok_dir}/grok_train_losses_opt{cfg.run_opt}", train_losses[0])
np.save(f"{grok_dir}/grok_train_acc_opt{cfg.run_opt}", train_acc[0])
np.save(f"{grok_dir}/grok_val_losses_opt{cfg.run_opt}", val_losses[0])
np.save(f"{grok_dir}/grok_val_acc_opt{cfg.run_opt}", val_acc[0])
