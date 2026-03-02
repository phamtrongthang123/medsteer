"""
Compute direction vectors from saved activation .pkl files.

Thin CLI wrapper around medsteer.directions.compute_directions().
"""

import argparse
import os

from medsteer.directions import compute_directions, save_directions, default_output_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute direction vectors from saved activation .pkl files."
    )
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory containing .pkl activation files")
    parser.add_argument("--raw_csv_path", type=str, required=True, help="Path to raw.csv (uuid -> short label text)")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for direction vector .pickle file")
    parser.add_argument("--concept_positive", type=str, default="dyed lifted polyps", help="Positive label")
    parser.add_argument("--concept_negative", type=str, default="normal cecum", help="Negative label")
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-XL-2-512x512", help="Base model ID (for filename)")
    parser.add_argument("--output_name", type=str, default=None, help="Optional output filename override (without extension)")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    direction_vectors = compute_directions(
        activations_dir=args.activations_dir,
        label_csv_path=args.raw_csv_path,
        concept_positive=args.concept_positive,
        concept_negative=args.concept_negative,
        model_id=args.model_id,
    )

    if args.output_name:
        filename = args.output_name + ".pickle"
    else:
        filename = default_output_filename(
            args.model_id, args.concept_positive, args.concept_negative
        )

    save_path = os.path.join(args.save_dir, filename)
    save_directions(direction_vectors, save_path)


if __name__ == "__main__":
    main()
