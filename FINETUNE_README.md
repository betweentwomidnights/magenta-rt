# Magenta RT Finetuning Scripts

Standalone Python scripts for finetuning Magenta RT models, extracted from the official Colab notebook.

## Overview

These scripts let you finetune Magenta RT on your own audio data using your DGX Spark (or any CUDA-enabled system).

**Two-step process:**
1. **Data Preparation** (`prepare_data.py`) - Tokenize audio, compute style embeddings
2. **Training** (`train.py`) - Finetune the model

## Prerequisites

All dependencies are included in the Docker image. If running locally, you need:
- Python 3.11
- JAX with CUDA support
- TensorFlow
- t5x, seqio, flaxformer
- magenta-realtime
- scikit-learn, matplotlib

## Quick Start

### Step 1: Prepare Your Audio Data

```bash
python prepare_data.py \
  --audio_dir /path/to/your/audio \
  --output_dir ./my_finetune \
  --task_name my_model
```

**What this does:**
- Finds all audio files (wav, mp3, flac, ogg)
- Tokenizes them using Magenta RT's featurizer
- Computes style embedding statistics (mean + 5 cluster centroids)
- Registers a SeqIO task for training
- Saves everything to `./my_finetune/`

**Outputs:**
- `my_model_examples.recordio` - Tokenized training data
- `mean_style_embed.npy` - Mean style embedding (for inference)
- `cluster_centroids.npy` - 5 cluster centroids (for inference)

### Step 2: Train the Model

```bash
python train.py \
  --task_name my_model \
  --output_dir ./my_finetune \
  --experiment_name dnb_20250107 \
  --model_size large \
  --num_steps 6000
```

**What this does:**
- Loads the prepared data
- Finetunes a Magenta RT model (base or large)
- Saves checkpoints every 1000 steps
- Generates training curves plot

**Outputs:**
- `dnb_20250107/checkpoint_1000/` - Checkpoint at step 1000
- `dnb_20250107/checkpoint_2000/` - Checkpoint at step 2000
- ... (every 1000 steps)
- `dnb_20250107/checkpoint_6000/` - Final checkpoint
- `dnb_20250107/training_curves.png` - Loss/accuracy plot

## Command Line Options

### prepare_data.py

```bash
python prepare_data.py --help

Required:
  --audio_dir PATH          Directory with your audio files

Optional:
  --output_dir PATH         Where to save outputs (default: ./mrt_finetune)
  --task_name NAME          SeqIO task name (default: audio_dir name)
  --audio_extensions LIST   File types to process (default: wav,mp3,flac,ogg)
  --filter_quiet            Filter out quiet segments
  --min_clip_seconds FLOAT  Min clip length (default: 2.0)
  --num_clusters INT        Number of centroids (default: 5)
```

### train.py

```bash
python train.py --help

Required:
  --task_name NAME          SeqIO task name from prepare_data.py

Optional:
  --output_dir PATH         Directory with prepared data (default: ./mrt_finetune)
  --experiment_name NAME    Training run name (default: timestamp)
  --model_size SIZE         base or large (default: large)
  --num_steps INT           Training steps (default: 6000)
  --save_period INT         Save checkpoint every N steps (default: 1000)
  --batch_size INT          Training batch size (default: 8)
```

## Example Workflows

### Minimal Example (drum & bass)

```bash
# 1. Prepare data
python prepare_data.py \
  --audio_dir /data/dnb_samples \
  --output_dir /outputs/dnb_finetune \
  --task_name dnb

# 2. Train (quick test - 2000 steps)
python train.py \
  --task_name dnb \
  --output_dir /outputs/dnb_finetune \
  --num_steps 2000
```

### Full Example (with all options)

```bash
# 1. Prepare data (filter quiet audio, only wav/flac)
python prepare_data.py \
  --audio_dir /data/guitar_collection \
  --output_dir /outputs/guitar_model \
  --task_name spanish_guitar \
  --audio_extensions wav,flac \
  --filter_quiet \
  --min_clip_seconds 3.0 \
  --num_clusters 8

# 2. Train (base model, longer training)
python train.py \
  --task_name spanish_guitar \
  --output_dir /outputs/guitar_model \
  --experiment_name guitar_jan2025 \
  --model_size base \
  --num_steps 10000 \
  --save_period 2000 \
  --batch_size 16
```

## Using with Docker

### Build the image:
```bash
docker build -t thecollabagepatch/magenta-finetune:latest .
```

### Run data preparation:
```bash
docker run --gpus all \
  -v /path/to/audio:/data \
  -v /path/to/outputs:/outputs \
  thecollabagepatch/magenta-finetune:latest \
  python prepare_data.py --audio_dir /data --output_dir /outputs
```

### Run training:
```bash
docker run --gpus all \
  -v /path/to/outputs:/outputs \
  thecollabagepatch/magenta-finetune:latest \
  python train.py --task_name <task> --output_dir /outputs
```

## Tips

**How much audio data?**
- Minimum: ~30 minutes of consistent style
- Recommended: 1-2 hours
- More diverse data = more diverse outputs

**Training steps?**
- Quick test: 2000 steps (~10-15 min on DGX Spark)
- Standard: 6000 steps (~30-45 min)
- Long training: 10000+ steps (watch for overfitting)

**Model size?**
- `base`: Faster training, less memory, good for testing
- `large`: Better quality, more compute (recommended for final models)

**Batch size?**
- Default (8): Works on most GPUs
- Increase if you have VRAM: 16, 32
- Decrease if OOM: 4, 2

## Output Files Explained

After both scripts complete, you'll have:

```
my_finetune/
├── my_model_examples.recordio       # Tokenized training data
├── mean_style_embed.npy             # For inference steering
├── cluster_centroids.npy            # For inference steering
└── experiment_20250107/             # Training outputs
    ├── checkpoint_1000/             # Checkpoints for inference
    ├── checkpoint_2000/
    ├── checkpoint_6000/
    └── training_curves.png          # Loss/accuracy over time
```

**For inference**, you need:
- A checkpoint directory (e.g., `checkpoint_6000/`)
- `mean_style_embed.npy`
- `cluster_centroids.npy`

## Troubleshooting

**"Task not found":**
- Make sure you ran `prepare_data.py` first
- Check that `--task_name` matches in both scripts

**"No audio files found":**
- Check your `--audio_dir` path
- Verify file extensions match `--audio_extensions`

**Out of memory:**
- Reduce `--batch_size` (try 4 or 2)
- Use `--model_size base` instead of large

**Training too slow:**
- Increase `--batch_size` if you have VRAM
- Make sure you're using GPU (check with `nvidia-smi`)

## Next Steps

After training, use your finetuned model with:
- The Magenta RT inference API
- Your existing gary/terry/jerry backends
- The Colab demo notebook (Step 4)

Load the checkpoint like this:
```python
from magenta_rt import system

mrt = system.MagentaRT(
    tag='large',  # or 'base'
    checkpoint_dir='/outputs/experiment_20250107/checkpoint_6000',
)

# Load style embeddings for steering
mean_embed = np.load('/outputs/mean_style_embed.npy')
centroids = np.load('/outputs/cluster_centroids.npy')
```

## License

Same as Magenta RT:
- Code: Apache 2.0
- Model weights: CC-BY 4.0