"""
Attention Modulator for Transformer2DModel (DiT).

This module implements the core logic for modulating a Diffusion Transformer by intercepting
latent features immediately after the Cross-Attention operation.

Mechanism:
1. Registration: The `attach_hooks` function (in hooks.py) monkey-patches the `forward` method
   of every `BasicTransformerBlock` in the model.
2. Interception: During the forward pass, after the `attn2` (Cross-Attention) layer finishes
   its computation but BEFORE the residual connection (hidden_states = attn_output + hidden_states),
   the `attn_output` is passed to the modulator.
3. Operation:
    - During Recording (`mode="record"`): The modulator saves the mean activation
      vector for that specific layer and denoising step.
    - During Suppression (`mode="suppress"`): The modulator subtracts the aligned component
      of the `attn_output` along the direction vector, steering away from a concept.
4. Normalization: To maintain stability, the modulated vector is re-normalized to match the
   original activation's norm.
"""

import abc
import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


class GuidanceModule(abc.ABC):
    """Abstract base class for cross-attention guidance modules."""

    def __init__(self):
        self._current_step = 0
        self._total_blocks = -1
        self._current_block = 0

    def reset_state(self):
        self._current_step = 0
        self._current_block = 0

    def _on_step_complete(self):
        return

    @abc.abstractmethod
    def process_activation(self, activation, block_idx: int):
        raise NotImplementedError

    def __call__(self, activation, block_idx: int):
        activation = self.process_activation(activation, block_idx)

        self._current_block += 1
        if self._current_block == self._total_blocks:
            self._current_block = 0
            self._on_step_complete()
            self._current_step += 1
        return activation


class AttentionModulator(GuidanceModule):
    """
    Stores per-step, per-layer 1D (length C) summaries; when modulation is enabled, it adds/subtracts
    the step/layer direction vector along the channel dimension and renormalizes to preserve the
    original norm. Stored vectors are means over batch/token dims (using only the second half of the
    batch), and direction vectors are computed as pos - neg from those summaries.

    Designed for Transformer2DModel (DiT) where blocks are indexed sequentially.

    Example (PixArt-XL-2):
    28 transformer blocks, each containing a Cross-Attention (attn2) layer.

    Modes:
        "passthrough" — no modulation, only records activations
        "suppress" — subtracts the aligned component gated by dot-product
        "record" — same as passthrough (records activations without modulation)
    """

    def __init__(
        self,
        direction_vectors=None,
        mode: str = "passthrough",
        suppress_scale: float = 2.0,
        device: str = "cpu",
    ):
        super().__init__()
        self._step_buffer = self._empty_buffer()
        self._activation_cache = defaultdict(dict)
        self.direction_vectors = direction_vectors
        self.mode = mode
        self.suppress_scale = suppress_scale
        self.device = device

    def reset_state(self):
        super().reset_state()
        self._step_buffer = self._empty_buffer()
        self._activation_cache = defaultdict(dict)

    @staticmethod
    def _empty_buffer():
        return {"blocks": []}

    def process_activation(self, activation, block_idx: int):
        # Apply modulation if direction vectors are available and mode requires it
        if self.mode == "suppress" and self.direction_vectors is not None:
            max_step = max(self.direction_vectors.keys())
            # Clamp to last available step
            num_step = (
                self._current_step
                if self._current_step in self.direction_vectors
                else max_step
            )
            if num_step > max_step:
                num_step = max_step

            if "blocks" in self.direction_vectors[num_step]:
                # Check if block_idx is valid for the stored vectors
                if block_idx < len(self.direction_vectors[num_step]["blocks"]):
                    direction_vector = self.direction_vectors[num_step]["blocks"][
                        block_idx
                    ]
                    direction_vector = (
                        torch.tensor(direction_vector).to(self.device).view(1, 1, -1)
                    )

                    # save current norm of activation components
                    norm = torch.norm(activation, dim=2, keepdim=True)

                    # computing dot products between activation components and direction vector
                    sim = torch.tensordot(
                        activation, direction_vector, dims=([2], [2])
                    ).view(activation.size()[0], activation.size()[1], 1)
                    # only suppress if dot product is positive
                    sim = torch.where(sim > 0, sim, 0)

                    activation = activation - (
                        self.suppress_scale * sim
                    ) * direction_vector.expand(
                        activation.size(0), activation.size(1), -1
                    )

                    # renormalize
                    activation = activation / (
                        torch.norm(activation, dim=2, keepdim=True) + 1e-8
                    )
                    activation = activation * norm

        # save activation for further computing direction vectors
        # If batch size > 1 (CFG), we take the second half (conditional part)
        if activation.shape[0] > 1:
            captured = (
                activation.detach()
                .cpu()
                .numpy()[len(activation) // 2 :]
                .mean(axis=0)
                .mean(axis=0)
            )
        else:
            captured = activation.detach().cpu().numpy().mean(axis=0).mean(axis=0)

        self._step_buffer["blocks"].append(captured)

        return activation

    def _on_step_complete(self):
        self._activation_cache[self._current_step] = self._step_buffer
        self._step_buffer = self._empty_buffer()
