"""
Guided Image Generation Script.

Applies pre-computed direction vectors to guide the generation process away
from specific medical concepts.

Modes:
- baseline: Standard generation using the provided prompt.
- suppress: Removes a concept. Used with a "disease" prompt + a direction
  vector for that same disease (e.g., Prompt="Polyp" + Vector="Polyp" -> Output=Normal).
"""

import argparse
import gc
import os

import torch

from medsteer.directions import load_directions
from medsteer.pipeline import MedSteerPipeline


def main():
    parser = argparse.ArgumentParser(description="MedSteer guided generation.")
    parser.add_argument("--model", type=str, default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to LoRA checkpoint directory"
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--direction_vectors",
        type=str,
        default=None,
        help="Path to direction vectors .pickle file",
    )
    parser.add_argument("--finding", type=str, default=None)
    parser.add_argument(
        "--direction_vectors_dir", type=str, default="direction_vectors"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "suppress"],
        default=None,
    )
    parser.add_argument("--num_denoising_steps", type=int, default=20)
    parser.add_argument("--suppress_scale", type=float, default=2)
    parser.add_argument("--save_dir", type=str, default="images")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--seeds_file", type=str, default=None)
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]

    # Auto-configure
    if args.finding:
        finding_slug = args.finding.replace(" ", "_")
        if args.direction_vectors is None:
            args.direction_vectors = os.path.join(
                args.direction_vectors_dir,
                f"{model_slug}_{args.finding}_None.pickle",
            )

        if args.mode == "baseline":
            if args.prompt is None:
                args.prompt = f"An endoscopic image of {args.finding}"
            args.save_dir = os.path.join(args.save_dir, finding_slug, "baseline")
        elif args.mode == "suppress":
            if args.prompt is None:
                args.prompt = f"An endoscopic image of {args.finding}"
            args.save_dir = os.path.join(
                args.save_dir,
                finding_slug,
                f"suppress_scale{args.suppress_scale}",
            )
    else:
        if args.prompt is None:
            args.prompt = "An endoscopic image of normal cecum"

    # Load pipeline
    pipeline = MedSteerPipeline.from_pretrained(
        model_id=args.model,
        lora_path=args.lora_path,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    if args.seeds_file:
        with open(args.seeds_file) as f:
            seeds = [int(line.strip()) for line in f if line.strip()][:10]
    else:
        seeds = [args.seed + i for i in range(args.num_images)]

    # Load direction vectors if needed
    direction_vectors = None
    if args.mode != "baseline" and args.direction_vectors:
        print(f"Loading direction vectors from: {args.direction_vectors}")
        direction_vectors = load_directions(args.direction_vectors)

    if args.mode == "baseline" or not direction_vectors:
        print(f"Running baseline generation for prompt: {args.prompt}")
        for current_seed in seeds:
            image = pipeline.generate(
                prompt=args.prompt,
                seed=current_seed,
                num_steps=args.num_denoising_steps,
                mode="baseline",
            )
            output_path = os.path.join(args.save_dir, f"orig_seed{current_seed}.png")
            image.save(output_path)
            print(f"Saved: {output_path}")
            del image
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print(f"Running {args.mode} generation for prompt: {args.prompt}")
        for current_seed in seeds:
            image = pipeline.generate(
                prompt=args.prompt,
                seed=current_seed,
                num_steps=args.num_denoising_steps,
                mode=args.mode,
                direction_vectors=direction_vectors,
                suppress_scale=args.suppress_scale,
            )
            output_path = os.path.join(args.save_dir, f"steered_seed{current_seed}.png")
            image.save(output_path)
            print(f"Saved: {output_path}")
            del image
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
