"""
Hook registration for Transformer2DModel (DiT).

Monkey-patches BasicTransformerBlock.forward to intercept cross-attention
outputs and route them through an AttentionModulator instance.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def attach_hooks(model, modulator):
    """
    Monkey-patch every BasicTransformerBlock in `model` so that the output of
    attn2 (cross-attention) is routed through `modulator` before the residual
    connection.

    Args:
        model: A Transformer2DModel (DiT) with `transformer_blocks`.
        modulator: An AttentionModulator (or any GuidanceModule subclass).
    """

    def _make_block_forward(block, block_idx):
        from diffusers.models.attention import _chunked_feed_forward

        def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning(
                        "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                    )

            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if block.norm_type == "ada_norm":
                norm_hidden_states = block.norm1(hidden_states, timestep)
            elif block.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block.norm1(
                        hidden_states,
                        timestep,
                        class_labels,
                        hidden_dtype=hidden_states.dtype,
                    )
                )
            elif block.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = block.norm1(hidden_states)
            elif block.norm_type == "ada_norm_continuous":
                norm_hidden_states = block.norm1(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            elif block.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block.scale_shift_table[None]
                    + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = block.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                raise ValueError("Incorrect norm used")

            if block.pos_embed is not None:
                norm_hidden_states = block.pos_embed(norm_hidden_states)

            cross_attention_kwargs = (
                cross_attention_kwargs.copy()
                if cross_attention_kwargs is not None
                else {}
            )
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            attn_output = block.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
                if block.only_cross_attention
                else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if block.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif block.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            if gligen_kwargs is not None:
                hidden_states = block.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if block.attn2 is not None:
                if block.norm_type == "ada_norm":
                    norm_hidden_states = block.norm2(hidden_states, timestep)
                elif block.norm_type in [
                    "ada_norm_zero",
                    "layer_norm",
                    "layer_norm_i2vgen",
                ]:
                    norm_hidden_states = block.norm2(hidden_states)
                elif block.norm_type == "ada_norm_single":
                    norm_hidden_states = hidden_states
                elif block.norm_type == "ada_norm_continuous":
                    norm_hidden_states = block.norm2(
                        hidden_states, added_cond_kwargs["pooled_text_emb"]
                    )
                else:
                    raise ValueError("Incorrect norm")

                if (
                    block.pos_embed is not None
                    and block.norm_type != "ada_norm_single"
                ):
                    norm_hidden_states = block.pos_embed(norm_hidden_states)

                attn_output = block.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )

                # --- MedSteer Hook ---
                attn_output = modulator(attn_output, block_idx)
                # ---------------------

                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if block.norm_type == "ada_norm_continuous":
                norm_hidden_states = block.norm3(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            elif not block.norm_type == "ada_norm_single":
                norm_hidden_states = block.norm3(hidden_states)

            if block.norm_type == "ada_norm_zero":
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            if block.norm_type == "ada_norm_single":
                norm_hidden_states = block.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if block._chunk_size is not None:
                ff_output = _chunked_feed_forward(
                    block.ff,
                    norm_hidden_states,
                    block._chunk_dim,
                    block._chunk_size,
                )
            else:
                ff_output = block.ff(norm_hidden_states)

            if block.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif block.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

        return forward

    block_count = 0
    if hasattr(model, "transformer_blocks"):
        for i, block in enumerate(model.transformer_blocks):
            block.forward = _make_block_forward(block, i)
            block_count += 1

    modulator._total_blocks = block_count
    print(f"Attached hooks to {block_count} transformer blocks.")
