import json
import torch
import numpy as np
import argparse
import random

# Import train_rev.py
import train_rev

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

  grok_dir = "ablations"

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
    "num_steps": "500000"
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
  model = gpt
  train_data = train_loader
  val_data = val_loader
  epoch_loss = []
  epoch_acc = []
  val_loss = []
  val_acc = []
  for epoch in range(config.epochs):
    if config.num_steps > 0:
        epoch_result = train.train_one_epoch(
            model,
            train_data,
            optimizer,
            scheduler,
            config,
            val_data=val_data,
            silent=True
        )
        epoch_loss.append(epoch_result[0])
        epoch_acc.append(epoch_result[1])
        val_loss.append(epoch_result[2])
        val_acc.append(epoch_result[3])
    else:
        epoch_result = train.train_one_epoch(
                  model,
                  train_data,
                  optimizer,
                  scheduler,
                  config,
                  val_data=val_data,
                  silent=True
              )
        epoch_loss.append(epoch_result[0])
        epoch_acc.append(epoch_result[1])
        val_result = train.validate(model, val_data, config)
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
  print("Train done")
  train_losses = epoch_loss
  train_acc = epoch_acc
  val_losses = val_loss
  val_acc = val_acc

  # Testing
  train.test_model(gpt, test_loader, config)

  # Save trained model,configs and losses
  with open(f"{grok_dir}/{config.exp_name}_config_opt{run_opt}.json", "w") as f:
      json.dump(vars(config), f, indent=4)

  torch.save(gpt.state_dict(), f"{grok_dir}/{config.exp_name}_opt{run_opt}.pth")

  with open(f"{grok_dir}/{config.exp_name}_losses_opt{run_opt}.json", "w") as f:
      json.dump({"train_losses": train_losses[0], "val_losses": val_losses}, f)

  np.save(f"{grok_dir}/grok_train_losses_opt{run_opt}", train_losses[0])
  np.save(f"{grok_dir}/grok_train_acc_opt{run_opt}", train_acc[0])
  np.save(f"{grok_dir}/grok_val_losses_opt{run_opt}", val_losses)
  np.save(f"{grok_dir}/grok_val_acc_opt{run_opt}", val_acc)
