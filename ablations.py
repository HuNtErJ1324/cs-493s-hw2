import json
import torch
import numpy as np
import argparse
import random

# Import train_abl.py
import train_abl as train

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed()

# Testing each optimizer
for i in range(1, 4+1):
  parser = argparse.ArgumentParser()
  config = parser.parse_args([])

  grok_dir = "ablation"

  # Setup
  args = {
    "mod": "97",
    "math_op": "/",
    "epochs": "1",
    "n_layer": "2",
    "n_head": "4",
    "n_embd": "128",
    "block_size": "32",
    "dropout": "0.0",
    "bias": "True",
    "learning_rate": "1e-3",
    "weight_decay": "1",
    "batch_size": "512",
    "eval_every": "2000",
    "num_steps": "1000000"
  }

  opt_map = {1: "adam", 2: "adamw", 3: "adagrad", 4: "rmsprop"}
  run_opt = i
  args["optim"] = opt_map[run_opt]

  config = train.parse_args(train.build_arg_input(**args))
  print("Config loaded. Using config")

  train_loader, val_loader, test_loader = train.load_dataset(config)
  print("Data loaded")

  # Create the transformer
  gpt, optimizer, scheduler = train.gen_gpt(config)
  print("GPT built")

  # Training
  print("Train started")
  # Updated to capture grokking_stats
  train_losses, train_acc, val_losses, val_acc, grokking_stats = train.train_model(
      gpt, train_loader, optimizer, scheduler, config, val_loader, silent=True)
  print("Train done")

  # Testing
  test_losses, test_acc = train.test_model(gpt, test_loader, config)

  # Save trained model, configs, losses and grokking stats
  with open(f"{grok_dir}/{config.exp_name}_config_opt{run_opt}.json", "w") as f:
      json.dump(vars(config), f, indent=4)

  torch.save(gpt.state_dict(), f"{grok_dir}/{config.exp_name}_opt{run_opt}.pth")

  # Save all results including grokking stats
  with open(f"{grok_dir}/{config.exp_name}_results_opt{run_opt}.json", "w") as f:
      json.dump({
          "train_losses": train_losses[0], 
          "train_acc": train_acc[0],
          "val_losses": val_losses, 
          "val_acc": val_acc,
          "test_losses": test_losses,
          "test_acc": test_acc,
          "grokking_stats": grokking_stats
      }, f)

  # Keep the existing individual metric files
  np.save(f"{grok_dir}/grok_train_losses_opt{run_opt}", train_losses[0])
  np.save(f"{grok_dir}/grok_train_acc_opt{run_opt}", train_acc[0])
  np.save(f"{grok_dir}/grok_val_losses_opt{run_opt}", val_losses)
  np.save(f"{grok_dir}/grok_val_acc_opt{run_opt}", val_acc)
  np.save(f"{grok_dir}/grok_test_losses_opt{run_opt}", test_losses)
  np.save(f"{grok_dir}/grok_test_acc_opt{run_opt}", test_acc)
  # Save the grokking stats separately
  np.save(f"{grok_dir}/grok_stats_opt{run_opt}", grokking_stats)
  
  # Print the grokking statistics for this optimizer
  print(f"\n=== GROKKING STATISTICS FOR {opt_map[run_opt].upper()} ===")
  print(f"Train error vanished: {grokking_stats['train_error_vanished']}")
  if grokking_stats['train_error_vanished']:
      print(f"  at step: {grokking_stats['train_vanish_step']}")
      print(f"Test error reached threshold: {grokking_stats['test_error_zero']}")
      if grokking_stats['test_error_zero']:
          print(f"  at step: {grokking_stats['test_error_zero_step']}")
          print(f"Grokking steps: {grokking_stats['grokking_steps']}")
      else:
          print("  No grokking detected within training period")
