"""
Activation recording for direction vector computation.

Generates images while capturing per-block, per-step cross-attention activations.
"""

import gc
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from medsteer.modulator import AttentionModulator
from medsteer.hooks import attach_hooks


class ActivationRecorder:
    """
    Wraps an AttentionModulator in record mode to capture cross-attention
    activations during image generation.

    Usage:
        recorder = ActivationRecorder(pipe, device="cuda")
        recorder.record_single("A photo of a polyp", seed=42, save_dir="activations/")
        recorder.record_batch("metadata.csv", "raw.csv", "activations/")
    """

    def __init__(self, pipeline, device="cuda"):
        """
        Args:
            pipeline: A PixArtAlphaPipeline instance (already loaded).
            device: Device string ("cuda" or "cpu").
        """
        self.pipeline = pipeline
        self.device = device
        self.modulator = AttentionModulator(device=device, mode="record")
        attach_hooks(pipeline.transformer, self.modulator)

    def record_single(
        self,
        prompt: str,
        seed: int,
        save_dir: str,
        label: str = None,
        base_name: str = None,
        extension: str = ".jpg",
        num_inference_steps: int = 20,
    ):
        """
        Generate a single image and save its activations.

        Args:
            prompt: Text prompt for generation.
            seed: Random seed.
            save_dir: Output directory.
            label: Optional short label stored as metadata in the .pkl.
            base_name: Base filename (without extension). Defaults to f"image_{seed}".
            extension: Image file extension.
            num_inference_steps: Number of denoising steps.
        """
        os.makedirs(save_dir, exist_ok=True)

        if base_name is None:
            base_name = f"image_{seed}"

        img_path = os.path.join(save_dir, f"{base_name}_{seed}{extension}")
        pkl_path = os.path.join(save_dir, f"{base_name}_{seed}.pkl")

        self.modulator.reset_state()

        with torch.no_grad():
            output = self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(self.device).manual_seed(seed),
                use_resolution_binning=False,
            )

        output.images[0].save(img_path)

        activation_data = dict(self.modulator._activation_cache)
        if label is not None:
            activation_data["_label"] = label

        with open(pkl_path, "wb") as f:
            pickle.dump(activation_data, f)

        del output
        gc.collect()
        torch.cuda.empty_cache()

        return img_path, pkl_path

    def record_batch(
        self,
        metadata_csv: str,
        raw_csv: str,
        save_dir: str,
        num_inference_steps: int = 20,
        num_images: int = None,
        base_seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        prompt_prefix: str = "An endoscopic image of ",
    ):
        """
        Generate images for all rows in metadata_csv, capturing activations.
        Resume-safe: skips rows where both image and .pkl already exist.

        Args:
            metadata_csv: Path to CSV with (file_name, text) columns.
            raw_csv: Path to CSV with short labels for metadata storage.
            save_dir: Output directory.
            num_inference_steps: Number of denoising steps.
            num_images: Max images to generate (default: all rows).
            base_seed: Base seed; image at index idx uses seed base_seed + idx.
            rank: Worker rank for distributed generation.
            world_size: Total number of workers.
            prompt_prefix: Prefix to strip from text to get label.
        """
        os.makedirs(save_dir, exist_ok=True)

        meta_df = pd.read_csv(metadata_csv)

        # Build uuid -> label mapping from raw.csv
        raw_df = pd.read_csv(raw_csv)
        uuid_to_label = {}
        for _, row in raw_df.iterrows():
            uuid = os.path.splitext(row["file_name"])[0]
            label = row["text"].replace(prompt_prefix, "").strip()
            uuid_to_label[uuid] = label

        total = num_images if num_images is not None else len(meta_df)
        skipped = 0

        for idx in tqdm(
            range(rank, total, world_size),
            desc="Generating images + capturing activations",
        ):
            row = meta_df.iloc[idx % len(meta_df)]
            file_name = row["file_name"]
            prompt = row["text"].strip()
            seed = base_seed + idx

            base_name, extension = os.path.splitext(file_name)
            img_path = os.path.join(save_dir, f"{base_name}_{seed}{extension}")
            pkl_path = os.path.join(save_dir, f"{base_name}_{seed}.pkl")

            # Resume-safe: skip if both outputs already exist
            if os.path.exists(img_path) and os.path.exists(pkl_path):
                skipped += 1
                continue

            label = uuid_to_label.get(base_name)

            self.modulator.reset_state()

            with torch.no_grad():
                output = self.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(self.device).manual_seed(seed),
                    use_resolution_binning=False,
                )

            output.images[0].save(img_path)

            activation_data = dict(self.modulator._activation_cache)
            if label is not None:
                activation_data["_label"] = label

            with open(pkl_path, "wb") as f:
                pickle.dump(activation_data, f)

            del output
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\nDone. Skipped (already exist): {skipped}")
        print(f"Images and activations saved to: {save_dir}")
