"""
MedSteerPipeline — unified model loading and generation interface.

Consolidates the repeated model-loading boilerplate from multiple scripts
into a single class with a clean API.
"""

import gc
import os

import torch
from peft import PeftModel
from transformers import T5EncoderModel

from diffusers import PixArtAlphaPipeline, Transformer2DModel
from medsteer.directions import load_directions
from medsteer.hooks import attach_hooks
from medsteer.modulator import AttentionModulator


class MedSteerPipeline:
    """
    Unified pipeline for MedSteer: model loading + guided generation.

    Usage:
        pipeline = MedSteerPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-512x512",
            lora_path="checkpoint-best-acc",
        )
        image = pipeline.generate("An endoscopic image of polyps", seed=42)

        # With direction-guided suppression:
        image = pipeline.generate(
            "An endoscopic image of dyed lifted polyps",
            seed=42,
            mode="suppress",
            direction_vectors_path="directions.pickle",
            suppress_scale=2.0,
        )"""

    def __init__(self, pipe, device="cuda"):
        """
        Args:
            pipe: A loaded PixArtAlphaPipeline.
            device: Device string.
        """
        self.pipe = pipe
        self.device = device
        self._modulator = None
        self._hooks_attached = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
        lora_path: str = None,
        device: str = "cuda",
        dtype=torch.float32,
    ):
        """
        Load PixArt-Alpha with optional LoRA adapters.

        Args:
            model_id: HuggingFace model ID or local path.
            lora_path: Path to LoRA checkpoint directory
                       (contains transformer_lora/ and text_encoder_lora/).
            device: Target device.
            dtype: Model dtype.

        Returns:
            MedSteerPipeline instance.
        """
        print(f"Loading model: {model_id}")
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        )
        transformer = Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype
        )

        if lora_path:
            print(f"Loading LoRA adapters from {lora_path}")
            transformer = PeftModel.from_pretrained(
                transformer, os.path.join(lora_path, "transformer_lora")
            )
            text_encoder = PeftModel.from_pretrained(
                text_encoder, os.path.join(lora_path, "text_encoder_lora")
            )

        pipe = PixArtAlphaPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=dtype,
        )
        pipe.vae.to(dtype=torch.float32)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        pipe.to(device)

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention.")
        except Exception as e:
            print(f"Could not enable xformers: {e}. Falling back to default attention.")

        return cls(pipe, device=device)

    def _ensure_hooks(self, modulator):
        """Attach hooks if not already attached, or if modulator changed."""
        if not self._hooks_attached or self._modulator is not modulator:
            self._modulator = modulator
            attach_hooks(self.pipe.transformer, modulator)
            self._hooks_attached = True

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        num_steps: int = 20,
        mode: str = "baseline",
        direction_vectors=None,
        direction_vectors_path: str = None,
        suppress_scale: float = 2.0,
    ):
        """
        Generate a single image.

        Args:
            prompt: Text prompt.
            seed: Random seed.
            num_steps: Number of denoising steps.
            mode: One of "baseline", "suppress".
            direction_vectors: Pre-loaded direction vectors dict, or None.
            direction_vectors_path: Path to .pickle file (loaded if direction_vectors is None).
            suppress_scale: Scale factor for suppress mode.

        Returns:
            PIL Image.
        """
        if mode != "baseline":
            if direction_vectors is None and direction_vectors_path is not None:
                direction_vectors = load_directions(direction_vectors_path)

            modulator = AttentionModulator(
                direction_vectors=direction_vectors,
                mode=mode,
                suppress_scale=suppress_scale,
                device=self.device,
            )
            self._ensure_hooks(modulator)
            modulator.reset_state()

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                use_resolution_binning=False,
            ).images[0]

        return result

    def generate_batch(
        self,
        prompts: list,
        seeds: list,
        num_steps: int = 20,
        mode: str = "baseline",
        direction_vectors=None,
        direction_vectors_path: str = None,
        suppress_scale: float = 2.0,
    ):
        """
        Generate multiple images sequentially with memory cleanup.

        Args:
            prompts: List of text prompts.
            seeds: List of random seeds (same length as prompts).
            num_steps: Number of denoising steps.
            mode: One of "baseline", "suppress".
            direction_vectors: Pre-loaded direction vectors dict.
            direction_vectors_path: Path to .pickle file.
            suppress_scale: Scale factor for suppress mode.

        Returns:
            List of PIL Images.
        """
        if direction_vectors is None and direction_vectors_path is not None:
            direction_vectors = load_directions(direction_vectors_path)

        images = []
        for prompt, seed in zip(prompts, seeds):
            image = self.generate(
                prompt=prompt,
                seed=seed,
                num_steps=num_steps,
                mode=mode,
                direction_vectors=direction_vectors,
                suppress_scale=suppress_scale,
            )
            images.append(image)
            gc.collect()
            torch.cuda.empty_cache()

        return images
