#!/usr/bin/env bash
# run_steer.sh — Single steering pass demo for MedSteer
# Usage:
#   bash run_steer.sh                    # default: suppress, dyed lifted polyps → normal cecum
#   bash run_steer.sh baseline           # no steering
#   bash run_steer.sh suppress           # suppress (polyp → normal)
#   bash run_steer.sh all                # both modes side-by-side

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
GITHUB_PUB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$GITHUB_PUB")"

MODEL="PixArt-alpha/PixArt-XL-2-512x512"
LORA_PATH="$REPO_DIR/medart_full_train_val_1040822/checkpoint-best-acc"
VECTORS_DIR="$REPO_DIR/medart_full_train_val_1040822/steering_vectors_raw"

# Concept pair (edit these to try other pairs)
POS_CONCEPT="dyed lifted polyps"
NEG_CONCEPT="normal cecum"
VECTOR_FILE="$VECTORS_DIR/PixArt-XL-2-512x512_${POS_CONCEPT}_${NEG_CONCEPT}.pickle"

# Generation settings
SEED=42
NUM_STEPS=20
SUPPRESS_SCALE=2.0

# Prompts
PROMPT_POS="An endoscopic image of ${POS_CONCEPT}"

# Output
SAVE_DIR="$REPO_DIR/steer_output"
mkdir -p "$SAVE_DIR"

# ─── Activate env ────────────────────────────────────────────────────────────
# If you need to activate conda: uncomment the next two lines
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate medart

export PYTHONPATH="$GITHUB_PUB${PYTHONPATH:+:$PYTHONPATH}"
cd "$GITHUB_PUB"

# ─── Run modes ───────────────────────────────────────────────────────────────
run_baseline() {
    local out_dir="$SAVE_DIR/baseline"
    mkdir -p "$out_dir"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Mode    : baseline"
    echo "  Prompt  : $PROMPT_POS"
    echo "  Seed    : $SEED"
    echo "  Output  : $out_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/generate.py \
        --model                "$MODEL" \
        --lora_path            "$LORA_PATH" \
        --prompt               "$PROMPT_POS" \
        --mode                 baseline \
        --seed                 "$SEED" \
        --num_images           1 \
        --num_denoising_steps  "$NUM_STEPS" \
        --save_dir             "$out_dir"
    echo "  Saved → $out_dir/orig_seed${SEED}.png"
}

run_suppress() {
    local out_dir="$SAVE_DIR/suppress"
    mkdir -p "$out_dir"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Mode    : suppress  (β=$SUPPRESS_SCALE)"
    echo "  Prompt  : $PROMPT_POS"
    echo "  Vector  : ${POS_CONCEPT} ↔ ${NEG_CONCEPT}"
    echo "  Seed    : $SEED"
    echo "  Output  : $out_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/generate.py \
        --model                "$MODEL" \
        --lora_path            "$LORA_PATH" \
        --prompt               "$PROMPT_POS" \
        --mode                 suppress \
        --direction_vectors    "$VECTOR_FILE" \
        --suppress_scale       "$SUPPRESS_SCALE" \
        --seed                 "$SEED" \
        --num_images           1 \
        --num_denoising_steps  "$NUM_STEPS" \
        --save_dir             "$out_dir"
    echo "  Saved → $out_dir/steered_seed${SEED}.png"
}

run_inject() {
    local out_dir="$SAVE_DIR/inject"
    mkdir -p "$out_dir"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Mode    : inject  (α=$INJECT_SCALE)"
    echo "  Prompt  : $PROMPT_NEG"
    echo "  Vector  : ${POS_CONCEPT} ↔ ${NEG_CONCEPT}"
    echo "  Seed    : $SEED"
    echo "  Output  : $out_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/generate.py \
        --model                "$MODEL" \
        --lora_path            "$LORA_PATH" \
        --prompt               "$PROMPT_NEG" \
        --mode                 inject \
        --direction_vectors    "$VECTOR_FILE" \
        --inject_scale         "$INJECT_SCALE" \
        --seed                 "$SEED" \
        --num_images           1 \
        --num_denoising_steps  "$NUM_STEPS" \
        --save_dir             "$out_dir"
    echo "  Saved → $out_dir/steered_seed${SEED}.png"
}

# ─── Main ────────────────────────────────────────────────────────────────────
MODE="${1:-suppress}"

case "$MODE" in
  baseline) run_baseline ;;
  suppress) run_suppress ;;
  inject)   run_inject ;;
  all)
    run_baseline
    run_suppress
    run_inject
    echo ""
    echo "All outputs in: $SAVE_DIR"
    echo "  $SAVE_DIR/baseline/orig_seed${SEED}.png"
    echo "  $SAVE_DIR/suppress/steered_seed${SEED}.png"
    echo "  $SAVE_DIR/inject/steered_seed${SEED}.png"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash run_steer.sh [baseline|suppress|inject|all]"
    exit 1
    ;;
esac

echo ""
echo "Done."
