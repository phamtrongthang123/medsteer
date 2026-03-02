"""
Capture cross-attention activations during image generation.

Generates images from metadata.csv while recording per-block, per-step
cross-attention activations for direction vector computation.
"""

import argparse
import os

import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from peft import PeftModel
from transformers import T5EncoderModel

from medsteer.capture import ActivationRecorder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images and capture cross-attention activations."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata.csv")
    parser.add_argument("--raw_csv_path", type=str, required=True, help="Path to raw.csv")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="LoRA checkpoint directory")
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0, help="Worker rank (0-based)")
    parser.add_argument("--world_size", type=int, default=1, help="Total workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    print(f"Loading model from {args.model_id} and LoRA from {args.ckpt_path}...")
    transformer = Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float32
    )
    transformer = PeftModel.from_pretrained(
        transformer, os.path.join(args.ckpt_path, "transformer_lora")
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=torch.float32
    )
    text_encoder = PeftModel.from_pretrained(
        text_encoder, os.path.join(args.ckpt_path, "text_encoder_lora")
    )

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    pipe.vae.to(dtype=torch.float32)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"Could not enable xformers: {e}.")

    # Create recorder and run
    recorder = ActivationRecorder(pipe, device=device)
    recorder.record_batch(
        metadata_csv=args.csv_path,
        raw_csv=args.raw_csv_path,
        save_dir=args.save_dir,
        num_inference_steps=args.num_inference_steps,
        num_images=args.num_images,
        base_seed=args.seed,
        rank=args.rank,
        world_size=args.world_size,
    )


if __name__ == "__main__":
    main()
