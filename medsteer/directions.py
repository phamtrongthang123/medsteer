"""
Direction vector computation and I/O.

Computes direction vectors from pre-captured activation .pkl files.
Loads activations, splits them into positive/negative groups by label, then computes:
  direction_vector[step][block] = normalize(mean(pos) - mean(neg))
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_directions(
    activations_dir: str,
    label_csv_path: str,
    concept_positive: str,
    concept_negative: str,
    model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
    prompt_prefix: str = "An endoscopic image of ",
):
    """
    Compute direction vectors from saved activation .pkl files.

    Args:
        activations_dir: Directory containing .pkl activation files.
        label_csv_path: Path to CSV with (file_name, text) for label lookup.
        concept_positive: Positive concept label (e.g., "dyed lifted polyps").
        concept_negative: Negative concept label (e.g., "normal cecum").
        model_id: Base model ID (used only for output filename generation).
        prompt_prefix: Prefix to strip from text column to get the label.

    Returns:
        dict: Direction vectors indexed as direction_vectors[step]["blocks"][block_idx].
    """
    # Load label CSV: uuid -> label
    raw_df = pd.read_csv(label_csv_path)
    uuid_to_label = {}
    for _, row in raw_df.iterrows():
        uuid = os.path.splitext(row["file_name"])[0]
        label = row["text"].replace(prompt_prefix, "").strip()
        uuid_to_label[uuid] = label

    # Scan activations_dir for all .pkl files
    all_pkl = sorted(
        f for f in os.listdir(activations_dir) if f.endswith(".pkl")
    )
    print(f"Found {len(all_pkl)} activation files.")

    pos_activations = []
    neg_activations = []
    skipped = 0

    for fname in tqdm(all_pkl, desc="Loading activations"):
        # Parse uuid from filename: {uuid}_{seed}.pkl
        base = fname[:-4]  # strip .pkl
        last_underscore = base.rfind("_")
        if last_underscore == -1:
            skipped += 1
            continue

        uuid = base[:last_underscore]

        label = uuid_to_label.get(uuid)
        if label is None:
            skipped += 1
            continue

        if label != concept_positive and label != concept_negative:
            continue

        pkl_path = os.path.join(activations_dir, fname)
        with open(pkl_path, "rb") as f:
            activation = pickle.load(f)

        if label == concept_positive:
            pos_activations.append(activation)
        else:
            neg_activations.append(activation)

    print(f"Positive activations: {len(pos_activations)}")
    print(f"Negative activations: {len(neg_activations)}")
    if skipped:
        print(f"Skipped (parse error or unknown uuid): {skipped}")

    if len(pos_activations) == 0:
        raise ValueError(
            f"No activations found for positive label: '{concept_positive}'"
        )
    if len(neg_activations) == 0:
        raise ValueError(
            f"No activations found for negative label: '{concept_negative}'"
        )

    # Compute direction vectors
    num_steps = sum(1 for k in pos_activations[0] if isinstance(k, int))
    print(f"Computing direction vectors for {num_steps} denoising steps.")

    direction_vectors = {}

    for denoising_step in range(num_steps):
        direction_vectors[denoising_step] = defaultdict(list)

        num_blocks = len(pos_activations[0][denoising_step]["blocks"])

        for block_idx in range(num_blocks):
            pos_layer = [
                pos_activations[i][denoising_step]["blocks"][block_idx]
                for i in range(len(pos_activations))
            ]
            pos_avg = np.mean(pos_layer, axis=0)

            neg_layer = [
                neg_activations[i][denoising_step]["blocks"][block_idx]
                for i in range(len(neg_activations))
            ]
            neg_avg = np.mean(neg_layer, axis=0)

            direction = pos_avg - neg_avg
            # L2-normalize
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm

            direction_vectors[denoising_step]["blocks"].append(direction)

    return direction_vectors


def save_directions(direction_vectors: dict, path: str):
    """Save direction vectors to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(direction_vectors, f)
    print(f"Saved direction vectors to {path}")


def load_directions(path: str) -> dict:
    """Load direction vectors from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def default_output_filename(
    model_id: str, concept_positive: str, concept_negative: str
) -> str:
    """Generate the default filename for direction vectors."""
    model_slug = model_id.split("/")[-1]
    return f"{model_slug}_{concept_positive}_{concept_negative}.pickle"
