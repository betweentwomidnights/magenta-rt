#!/usr/bin/env python3
"""
Train (finetune) a Magenta RT model on prepared data.

This script:
1. Loads the registered SeqIO task
2. Sets up the Magenta RT finetuner
3. Trains for specified number of steps
4. Saves checkpoints periodically
5. Plots training curves
"""

import argparse
import os
import pathlib
from datetime import datetime
import seqio
import t5x.utils
import clu.data
import tensorflow.data as tf_data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt

from magenta_rt.finetune import finetuner, tasks


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune Magenta RT model')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of the registered SeqIO task')
    parser.add_argument('--output_dir', type=str, default='./mrt_finetune',
                        help='Directory containing prepared data and to save outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this training run (defaults to timestamp)')
    parser.add_argument('--model_size', type=str, default='large', choices=['base', 'large'],
                        help='Model size to finetune')
    parser.add_argument('--num_steps', type=int, default=6000,
                        help='Number of training steps')
    parser.add_argument('--save_period', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (uses default if not specified)')
    return parser.parse_args()


def get_dataset_iterator(task_name: str, batch_size: int = 8):
    """Create a dataset iterator for training."""
    train_dataset_cfg = t5x.utils.DatasetConfig(
        mixture_or_task_name=task_name,
        task_feature_lengths={'inputs': 1006, 'targets': 800},
        split='train',
        batch_size=batch_size,
        shuffle=True,
        use_cached=False,
        pack=True,
        module=None,
        seed=42,
    )
    
    train_ds = t5x.utils.get_dataset(
        cfg=train_dataset_cfg,
        shard_id=0,
        num_shards=1,
        feature_converter_cls=seqio.EncDecFeatureConverter,
    )
    
    train_iter = clu.data.dataset_iterator.TfDatasetIterator(train_ds, checkpoint=False)
    return train_iter


def plot_training_curves(training_summary: dict, 
                          save_path: pathlib.Path,
                          experiment_name: str):
    """Plot and save training curves."""
    num_plots = len(training_summary.keys())
    fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    # Handle single plot case
    if num_plots == 1:
        axs = [axs]
    
    fig.suptitle(f"Training curves for {experiment_name}")
    
    for i, (k, v) in enumerate(training_summary.items()):
        axs[i].plot([item.value for item in v])
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel(k)
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")


def verify_task_exists(task_name: str, output_dir: pathlib.Path):
    """Verify the SeqIO task exists and has data. Register it if needed."""
    # Check for recordio file
    recordio_path = output_dir / f'{task_name}_examples.recordio'
    if not recordio_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {recordio_path}\n"
            f"Did you run prepare_data.py with --output_dir {output_dir}?"
        )
    
    # Register task if it doesn't exist
    if task_name not in seqio.TaskRegistry.names():
        print(f"Registering SeqIO task '{task_name}'...")
        
        # Remove if it somehow exists
        try:
            seqio.TaskRegistry.remove(task_name)
            seqio.TaskRegistry.remove(task_name + "_eval")
        except:
            pass
        
        tasks.register_task(
            name=task_name,
            split_to_filepattern={
                'train': str(recordio_path),
                'validation': str(recordio_path),
            },
            reader_cls=tf_data.TFRecordDataset,
            acoustic_key='acoustic_tokens',
            style_key='style_tokens',
            encoder_codec_rvq_depth=4,
            decoder_codec_rvq_depth=16,
            max_prompt_secs=10,
        )
        print(f"✓ Task '{task_name}' registered")
    else:
        print(f"✓ Task '{task_name}' already registered")


def main():
    args = parse_args()
    
    # Setup paths
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory {output_dir} does not exist.\n"
            f"Did you run prepare_data.py first?"
        )
    
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M")
    model_output_dir = output_dir / experiment_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment name: {experiment_name}")
    print(f"Model size: {args.model_size}")
    print(f"Output directory: {model_output_dir}")
    print(f"Training steps: {args.num_steps}")
    print(f"Save checkpoint every: {args.save_period} steps")
    
    # Verify task exists (and register if needed)
    verify_task_exists(args.task_name, output_dir)
    
    # Setup finetuner
    print("\nInitializing Magenta RT finetuner...")
    trainer = finetuner.MagentaRTFinetuner(
        tag=args.model_size,
        output_dir=str(model_output_dir),
    )
    
    # Create dataset iterator
    print("Loading training data...")
    train_iter = get_dataset_iterator(args.task_name, batch_size=args.batch_size)
    
    # Train
    print(f"\n🚀 Starting training for {args.num_steps} steps...")
    print("=" * 80)
    
    trainer.train(
        train_iter=train_iter,
        num_steps=args.num_steps,
        save_ckpt_period=args.save_period,
    )
    
    print("=" * 80)
    print("✅ Training complete!")
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(
        training_summary={
            'Loss': trainer.loss,
            'Accuracy': trainer.accuracy,
        },
        save_path=model_output_dir / 'training_curves.png',
        experiment_name=experiment_name
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Model size: {args.model_size}")
    print(f"Steps completed: {args.num_steps}")
    print(f"Final loss: {trainer.loss[-1].value:.4f}")
    print(f"Final accuracy: {trainer.accuracy[-1].value:.4f}")
    print(f"\nCheckpoints saved to: {model_output_dir}")
    
    # List saved checkpoints
    checkpoints = sorted([d for d in model_output_dir.iterdir() if d.name.startswith('checkpoint_')])
    if checkpoints:
        print(f"\nAvailable checkpoints:")
        for ckpt in checkpoints:
            print(f"  - {ckpt.name}")
    
    print("\n" + "=" * 80)
    print(f"\nTo use this model for inference, load from: {model_output_dir}")


if __name__ == '__main__':
    main()