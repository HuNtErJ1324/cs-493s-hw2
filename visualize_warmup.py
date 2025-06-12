import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_experiment_curves(base_path, eval_every, save_path=None):
    """
    Loads training and validation data based on a base file path and plots the curves
    against the number of training steps.

    Args:
        base_path (str): The base path for the experiment files, without the metric suffix or extension.
                         Example: 'warmup/warmup_1_97_/_1'
        eval_every (int): The frequency of validation (e.g., validation is run every 1000 steps).
        save_path (str, optional): Path to save the generated plot image. If None, displays the plot.
    """
    try:
        # Construct full paths to the .npy files
        train_loss_path = f"{base_path}_train_losses.npy"
        val_loss_path = f"{base_path}_val_losses.npy"
        train_acc_path = f"{base_path}_train_acc.npy"
        val_acc_path = f"{base_path}_val_acc.npy"

        print(f"Loading data from: {base_path}_*.npy")
        
        # Load the data from .npy files
        train_loss = np.load(train_loss_path)
        val_loss = np.load(val_loss_path)
        train_acc = np.load(train_acc_path)
        val_acc = np.load(val_acc_path)
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]
        print(f"Final Training Accuracy:  {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file.")
        print(f"Missing file: {e.filename}")
        print("Please ensure the training script ran successfully and the path is correct.")
        return

    train_steps = range(1, len(train_loss) + 1)
    
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Experiment Results for: {os.path.basename(base_path)}', fontsize=16)

    # Loss curves 
    ax_loss.plot(train_steps, train_loss, label='Training Loss', alpha=0.7)
    ax_loss.plot(train_steps, np.repeat(val_loss, 1000), label='Validation Loss')
    ax_loss.set_title('Training & Validation Loss', fontsize=14)
    ax_loss.set_xlabel('Training Steps (log scale)', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_xscale('log') 
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.6)

    # Acc curves 
    ax_acc.plot(train_steps, train_acc, label='Training Accuracy', alpha=0.7)
    ax_acc.plot(train_steps, np.repeat(val_acc, 1000), label='Validation Accuracy')
    ax_acc.set_title('Training & Validation Accuracy', fontsize=14)
    ax_acc.set_xlabel('Training Steps (log scale)', fontsize=12)
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.set_xscale('log') 
    ax_acc.legend()
    ax_acc.grid(True, linestyle='--', alpha=0.6)
   
   # Adjust layout and show/save the plot
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    if save_path:
        # Ensure the save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    seeds = [1, 2, 3]
    ps = [97, 113]
    ops = ["+", "-"]
    ns = [1, 2] 
    
    for seed in seeds:
        for p in ps:
            for op in ops:
                for n in ns:
                    base_filename = f"warmup/warmup_{seed}_{p}_{op}_{n}"
                    save_path = f"warmup_viz/warmup_{seed}_{p}_{op}_{n}"

                    plot_experiment_curves(base_filename, 1000, save_path)

