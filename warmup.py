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

# Defining params for train runs
seeds = [1, 2, 3]
ps = [97, 113]
ns = [1, 2]
ops= ["+", "-"]

for seed in seeds:
    for p in ps:
        for n in ns:
            for op in ops:
                # Setting random seed based on config
                set_seed(seed)
                print(f"Seed set to: {seed}")

                warmup_dir = "warmup"

                # Setup 
                args = {
                    'mod': str(p),
                    "math_op": op,
                    "epochs": "1",
                    "n_layer": str(n),
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

                save_string = f"warmup/{config.exp_name}_{str(seed)}_{str(p)}_{op}_{str(n)}"

                # Save trained model,configs and losses
                with open(f"{save_string}_config.json", "w") as f:
                    json.dump(vars(config), f, indent=4)

                torch.save(gpt.state_dict(), f"{save_string}.pth")

                with open(f"{save_string}_losses.json", "w") as f:
                    json.dump({"train_losses": train_losses[0], "val_losses": val_losses}, f)

                np.save(f"{save_string}_train_losses", train_losses[0])
                np.save(f"{save_string}_train_acc", train_acc[0])
                np.save(f"{save_string}_val_losses", val_losses[0])
                np.save(f"{save_string}_val_acc", val_acc[0])
