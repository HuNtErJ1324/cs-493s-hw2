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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Setting up arguments for different experiments
parser = argparse.ArgumentParser(description="Run a training experiment with specific configurations.")
parser.add_argument("--seed", type=int, default=1, choices=[1, 2, 3], help="Random seed for reproducibility (choices: 1, 2, 3).")
parser.add_argument("--mod", type=int, default=97, choices=[97, 113], help="Modulo prime for the mathematical operation (choices: 97, 113).")
parser.add_argument("--n_layer", type=int, default=1, choices=[1, 2], help="Number of transformer layers (choices: 1, 2).")
parser.add_argument("--op", type=str, default="/", choices=["/", "+", "-"], help="Mathematical operation to perform (choices: '/', '+', '-').")
config_args = parser.parse_args()

# Setting random seed based on config
set_seed(config_args.seed)
print(f"Seed set to: {config_args.seed}")

warmup_dir = "warmup"

# Setup 
args = {
    'mod': str(config_args.mod),
    "math_op": config_args.op,
    "epochs": "1",
    "n_layer": str(config_args.n_layer),
    "n_head": "4",
    "n_embd": "128",
    "block_size": "32",  
    "dropout": "0.0",
    "bias": "True",
    "learning_rate": "1e-3",
    "weight_decay": "1",
    "batch_size": "512",
    'exp_name': "warmup",
    "eval_every": "1000",
    "num_steps": "100_000",
}

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
test_losses, test_acc = train.test_model(gpt, test_loader, config)

# Save trained model,configs and losses
with open(f"{warmup_dir}/{config.exp_name}_{str(config_args.seed)}_{str(config_args.mod)}_{config_args.op}_{str(config_args.n_layer)}_config.json", "w") as f:
    json.dump(vars(config), f, indent=4)

torch.save(gpt.state_dict(), f"{warmup_dir}/{config.exp_name}_{str(config_args.seed)}_{str(config_args.mod)}_{config_args.op}_{str(config_args.n_layer)}.pth")

with open(f"{warmup_dir}/{config.exp_name}_losses_opt{config.run_opt}_{str(config_args.seed)}_{str(config_args.mod)}_{config_args.op}_{str(config_args.n_layer)}_losses.json", "w") as f:
    json.dump({"train_losses": train_losses[0], "val_losses": val_losses}, f)


#np.save(f"{grok_dir}/grok_train_losses_opt{config.run_opt}", train_losses[0])
#np.save(f"{grok_dir}/grok_train_acc_opt{config.run_opt}", train_acc[0])
#np.save(f"{grok_dir}/grok_val_losses_opt{config.run_opt}", val_losses)
#np.save(f"{grok_dir}/grok_val_acc_opt{config.run_opt}", val_acc)
