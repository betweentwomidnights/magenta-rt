#!/usr/bin/env python3
"""
Prepare audio data for Magenta RT finetuning.

This script:
1. Loads audio files from a directory
2. Tokenizes them using the Magenta RT Featurizer
3. Writes TFRecord files for training
4. Computes style embedding statistics (mean + cluster centroids)
5. Registers a SeqIO task for training
"""

import argparse
import os
import pathlib
import numpy as np
import seqio
import tensorflow as tf
import tensorflow.data as tf_data
from tqdm import tqdm
from sklearn.cluster import KMeans

from magenta_rt.finetune import data, tasks
from magenta_rt import audio as audio_lib


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare audio data for Magenta RT finetuning')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='./mrt_finetune',
                        help='Directory to save outputs')
    parser.add_argument('--task_name', type=str, default=None,
                        help='Name for the SeqIO task (defaults to audio_dir name)')
    parser.add_argument('--audio_extensions', type=str, default='wav,mp3,flac,ogg',
                        help='Comma-separated list of audio extensions to process')
    parser.add_argument('--filter_quiet', action='store_true',
                        help='Filter out quiet audio segments')
    parser.add_argument('--min_clip_seconds', type=float, default=2.0,
                        help='Minimum clip length in seconds')
    parser.add_argument('--num_clusters', type=int, default=5,
                        help='Number of cluster centroids to compute')
    return parser.parse_args()


def find_audio_files(audio_dir: pathlib.Path, extensions: list[str]) -> list[pathlib.Path]:
    """Find all audio files in directory with given extensions."""
    audio_paths = []
    for ext in extensions:
        audio_paths.extend(list(audio_dir.glob(f'**/*.{ext}')))
    return sorted(audio_paths)


def tokenize_audio(audio_paths: list[pathlib.Path], 
                   output_pattern: str,
                   featurizer: data.Featurizer) -> int:
    """Tokenize audio files and write to TFRecord format."""
    records_count = 0
    with tf.io.TFRecordWriter(output_pattern) as file_writer:
        for audio_path in tqdm(audio_paths, desc="Tokenizing audio"):
            try:
                audio_input = audio_lib.Waveform.from_file(audio_path)
                tokenized_iter = featurizer.process(audio_input)
                for tokenized_example in tokenized_iter:
                    records_count += 1
                    file_writer.write(tokenized_example.SerializeToString())
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    return records_count


def compute_style_statistics(recordio_path: str, 
                              num_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean embedding and cluster centroids from style embeddings."""
    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            record_bytes,
            {"style_embeddings": tf.io.FixedLenFeature([], dtype=tf.string)}
        )
    
    print("Computing style embedding statistics...")
    audio_style_embeddings = []
    dataset = tf_data.TFRecordDataset([recordio_path]).map(decode_fn)
    
    for batch in tqdm(dataset, desc="Processing embeddings"):
        style_embeds = tf.io.parse_tensor(batch['style_embeddings'], out_type=tf.float32).numpy()
        audio_style_embeddings.append(np.mean(style_embeds, axis=0))
    
    audio_style_embeddings = np.array(audio_style_embeddings)
    
    # Compute mean
    mean_style_embed = np.mean(audio_style_embeddings, axis=0)
    
    # Compute cluster centroids
    print(f"Computing {num_clusters} cluster centroids...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(audio_style_embeddings)
    cluster_centroids = kmeans.cluster_centers_
    
    return mean_style_embed, cluster_centroids


def main():
    args = parse_args()
    
    # Setup paths
    audio_dir = pathlib.Path(args.audio_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory {audio_dir} does not exist")
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_name = args.task_name or audio_dir.stem
    output_pattern = str(output_dir / f'{task_name}_examples.recordio')
    
    print(f"Task name: {task_name}")
    print(f"Output directory: {output_dir}")
    
    # Find audio files
    audio_extensions = [e.strip() for e in args.audio_extensions.split(',') if e.strip()]
    audio_paths = find_audio_files(audio_dir, audio_extensions)
    
    if not audio_paths:
        raise FileNotFoundError(
            f"No audio files found in {audio_dir} with extensions {audio_extensions}"
        )
    
    print(f"Found {len(audio_paths)} audio files")
    print(f"First few files:")
    for p in audio_paths[:5]:
        print(f"  - {p}")
    
    # Tokenize audio
    print("\nTokenizing audio files...")
    featurizer = data.Featurizer(
        filter_quiet=args.filter_quiet,
        min_clip_seconds=args.min_clip_seconds,
        include_style_embeddings=True,
    )
    
    records_count = tokenize_audio(audio_paths, output_pattern, featurizer)
    
    print(f"\n{records_count} records written to {output_pattern}")
    featurized_audio_length = records_count * 30
    print(f"Total duration: {featurized_audio_length:.0f} seconds ({featurized_audio_length/60:.1f} minutes)")
    
    # Register SeqIO task
    print("\nRegistering SeqIO task...")
    if task_name in seqio.TaskRegistry.names():
        seqio.TaskRegistry.remove(task_name)
        seqio.TaskRegistry.remove(task_name + "_eval")
    
    tasks.register_task(
        name=task_name,
        split_to_filepattern={
            'train': output_pattern,
            'validation': output_pattern,
        },
        reader_cls=tf_data.TFRecordDataset,
        acoustic_key='acoustic_tokens',
        style_key='style_tokens',
        encoder_codec_rvq_depth=4,
        decoder_codec_rvq_depth=16,
        max_prompt_secs=10,
    )
    print(f"SeqIO task '{task_name}' registered")
    
    # Compute style statistics
    mean_style_embed, cluster_centroids = compute_style_statistics(
        output_pattern, 
        args.num_clusters
    )
    
    # Save outputs
    embeddings_path = output_dir / f'{task_name}_style_embeddings.npy'
    mean_path = output_dir / 'mean_style_embed.npy'
    centroids_path = output_dir / 'cluster_centroids.npy'
    
    # Note: we save task_name_style_embeddings for reference but don't actually need it for training
    # The mean and centroids are what matter
    print(f"\nSaving style statistics...")
    np.save(mean_path, mean_style_embed)
    np.save(centroids_path, cluster_centroids)
    
    print(f"  Mean embedding: {mean_path}")
    print(f"  Cluster centroids: {centroids_path}")
    
    print("\n✅ Data preparation complete!")
    print(f"\nTo train, run:")
    print(f"  python train.py --task_name {task_name} --output_dir {output_dir}")


if __name__ == '__main__':
    main()