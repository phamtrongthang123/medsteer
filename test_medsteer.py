#!/usr/bin/env python
"""
Test suite for the medsteer package.

Usage:
    python test_medsteer.py          # Levels 1+2 only (CPU, ~15s)
    python test_medsteer.py --gpu    # All levels including GPU integration (~3min)
"""

import argparse
import importlib
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths (relative to this file's parent directory — the medart repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_ROOT = os.path.dirname(REPO_ROOT)  # medart repo root

STEERING_VECTOR_FILE = os.path.join(
    PARENT_ROOT,
    "medart_full_train_val_1040822",
    "steering_vectors_raw",
    "PixArt-XL-2-512x512_dyed lifted polyps_normal cecum.pickle",
)
LORA_PATH = os.path.join(
    PARENT_ROOT, "medart_full_train_val_1040822", "checkpoint-best-acc"
)
MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
_results = {"pass": 0, "fail": 0, "skip": 0}


def run_test(name, fn, *, skip_if=False, skip_reason=""):
    """Execute *fn* and print [PASS]/[FAIL]/[SKIP]."""
    if skip_if:
        print(f"  [SKIP] {name} — {skip_reason}")
        _results["skip"] += 1
        return
    try:
        fn()
        print(f"  [PASS] {name}")
        _results["pass"] += 1
    except Exception as e:
        print(f"  [FAIL] {name} — {e}")
        traceback.print_exc()
        _results["fail"] += 1


# ===================================================================
# Level 1: Import & Structure (no GPU)
# ===================================================================


def test_top_level_imports():
    import medsteer

    expected = [
        "GuidanceModule",
        "AttentionModulator",
        "attach_hooks",
        "compute_directions",
        "load_directions",
        "save_directions",
        "ActivationRecorder",
        "MedSteerPipeline",
        "color_distribution_loss",
    ]
    for name in expected:
        assert hasattr(medsteer, name), f"medsteer.{name} not found"


def test_classifier_submodule_imports():
    from medsteer.classifier import KVASIR_LABELS

    assert len(KVASIR_LABELS) == 8, f"Expected 8 labels, got {len(KVASIR_LABELS)}"


def test_evaluation_submodule_imports():
    from medsteer.evaluation import compute_fid, frechet_distance
    from medsteer.evaluation.classifier_eval import evaluate_generated_images

    assert callable(frechet_distance)
    assert callable(compute_fid)
    assert callable(evaluate_generated_images)


def test_visualization_submodule_imports():
    from medsteer.visualization import make_grid
    from medsteer.visualization.comparison_grid import abs_diff_heatmap, extract_seed

    assert callable(make_grid)
    assert callable(extract_seed)
    assert callable(abs_diff_heatmap)


def test_script_imports():
    """All 7 lightweight scripts importable via importlib."""
    lightweight = [
        "scripts.generate",
        "scripts.capture_activations",
        "scripts.compute_directions",
        "scripts.sample_batch",
        "scripts.evaluate_fid",
        "scripts.evaluate_classifier",
        "scripts.merge_grid",
    ]
    for mod_name in lightweight:
        m = importlib.import_module(mod_name)
        assert m is not None, f"Failed to import {mod_name}"


def test_train_script_imports():
    """train.py and train_val.py importable (heavier)."""
    for mod_name in ["scripts.train", "scripts.train_val"]:
        m = importlib.import_module(mod_name)
        assert m is not None, f"Failed to import {mod_name}"


def test_no_old_names_leak():
    """Old names (VectorStore, VectorControl, etc.) must not exist."""
    import medsteer
    import medsteer.hooks
    import medsteer.modulator

    old_names = [
        "VectorStore",
        "VectorControl",
        "CASteer",
        "steer_forward",
        "steer_backward",
        "register_vector_control",
    ]
    for mod in [medsteer, medsteer.modulator, medsteer.hooks]:
        for name in old_names:
            assert not hasattr(mod, name), (
                f"Old name '{name}' leaked into {mod.__name__}"
            )


# ===================================================================
# Level 2: Unit Logic (CPU tensors, no GPU)
# ===================================================================


def test_modulator_passthrough():
    from medsteer.modulator import AttentionModulator

    mod = AttentionModulator(mode="passthrough")
    mod._total_blocks = 1  # simulate 1 block so stepping works
    x = torch.randn(2, 1024, 1152)
    out = mod(x.clone(), block_idx=0)
    assert torch.allclose(out, x), "Passthrough should return input unchanged"


def test_modulator_passthrough_records_cache():
    from medsteer.modulator import AttentionModulator

    mod = AttentionModulator(mode="passthrough")
    mod._total_blocks = 2
    x = torch.randn(2, 1024, 1152)
    # Run through 2 blocks to complete one step
    mod(x.clone(), block_idx=0)
    mod(x.clone(), block_idx=1)
    assert 0 in mod._activation_cache, "_activation_cache should have step 0"
    blocks = mod._activation_cache[0]["blocks"]
    assert len(blocks) == 2, f"Expected 2 block captures, got {len(blocks)}"
    assert blocks[0].shape == (1152,), f"Expected shape (1152,), got {blocks[0].shape}"


def test_modulator_suppress():
    from medsteer.modulator import AttentionModulator

    # Build a simple direction vector: step=0, block=0
    direction = np.random.randn(1152).astype(np.float32)
    direction /= np.linalg.norm(direction)
    dv = {0: {"blocks": [direction]}}
    mod = AttentionModulator(
        direction_vectors=dv, mode="suppress", suppress_scale=2.0, device="cpu"
    )
    mod._total_blocks = 1

    x = torch.randn(1, 64, 1152)
    orig_norm = torch.norm(x, dim=2, keepdim=True)
    out = mod(x.clone(), block_idx=0)
    out_norm = torch.norm(out, dim=2, keepdim=True)
    # Norm should be preserved (renormalized)
    assert torch.allclose(orig_norm, out_norm, atol=1e-4), (
        "Suppress should preserve norm"
    )
    # Output should differ from input (direction was suppressed)
    assert not torch.allclose(out, x, atol=1e-6), (
        "Suppress should modify the activation"
    )


def test_modulator_stepping_logic():
    from medsteer.modulator import AttentionModulator

    mod = AttentionModulator(mode="passthrough")
    mod._total_blocks = 3
    assert mod._current_step == 0
    assert mod._current_block == 0

    x = torch.randn(1, 64, 1152)
    mod(x.clone(), block_idx=0)
    assert mod._current_block == 1
    mod(x.clone(), block_idx=1)
    assert mod._current_block == 2
    mod(x.clone(), block_idx=2)
    # After 3 blocks, step should advance
    assert mod._current_step == 1
    assert mod._current_block == 0

    mod.reset_state()
    assert mod._current_step == 0
    assert mod._current_block == 0
    assert len(mod._activation_cache) == 0


def test_modulator_suppress_negative_dot():
    """Negative dot product → no modification (gating)."""
    from medsteer.modulator import AttentionModulator

    direction = np.array([1.0] + [0.0] * 1151, dtype=np.float32)
    dv = {0: {"blocks": [direction]}}
    mod = AttentionModulator(
        direction_vectors=dv, mode="suppress", suppress_scale=2.0, device="cpu"
    )
    mod._total_blocks = 1

    # Input that is anti-aligned with the direction (negative dot)
    x = torch.zeros(1, 1, 1152)
    x[0, 0, 0] = -1.0
    x[0, 0, 1] = 1.0  # give it some norm in another dim
    out = mod(x.clone(), block_idx=0)
    # Since dot product is negative, gating zeros it out → norm-preserving identity
    orig_norm = torch.norm(x, dim=2, keepdim=True)
    out_norm = torch.norm(out, dim=2, keepdim=True)
    assert torch.allclose(orig_norm, out_norm, atol=1e-4), "Norm should be preserved"
    # The output should effectively be the same direction (no suppression applied)
    cosine = torch.nn.functional.cosine_similarity(x.view(1, -1), out.view(1, -1))
    assert cosine.item() > 0.99, (
        f"Expected cosine ~1 for anti-aligned input, got {cosine.item()}"
    )


def test_modulator_step_clamping():
    """Step > max_step clamps without crash."""
    from medsteer.modulator import AttentionModulator

    direction = np.random.randn(1152).astype(np.float32)
    direction /= np.linalg.norm(direction)
    # Only define step 0
    dv = {0: {"blocks": [direction]}}
    mod = AttentionModulator(
        direction_vectors=dv, mode="suppress", suppress_scale=2.0, device="cpu"
    )
    mod._total_blocks = 1
    mod._current_step = 99  # way beyond defined steps

    x = torch.randn(1, 64, 1152)
    # Should not crash — clamps to max_step=0
    out = mod(x.clone(), block_idx=0)
    assert out.shape == x.shape


def test_modulator_cfg_batch_capture():
    """Batch_size > 1 captures only conditional half (second half)."""
    from medsteer.modulator import AttentionModulator

    mod = AttentionModulator(mode="passthrough")
    mod._total_blocks = 1

    batch = torch.randn(4, 64, 1152)  # 4 = 2 unconditional + 2 conditional
    mod(batch.clone(), block_idx=0)

    # The captured activation should be mean of the second half (indices 2,3)
    expected = batch[2:].detach().cpu().numpy().mean(axis=0).mean(axis=0)
    captured = mod._activation_cache[0]["blocks"][0]
    assert np.allclose(expected, captured, atol=1e-5), (
        "Should capture conditional half only"
    )


def test_color_distribution_loss():
    from medsteer.losses import color_distribution_loss

    x = torch.randn(2, 3, 64, 64)
    # Identical → ~0
    loss_same = color_distribution_loss(x, x)
    assert loss_same.item() < 1e-6, (
        f"Identical images loss should be ~0, got {loss_same.item()}"
    )
    # Different → > 0
    y = torch.randn(2, 3, 64, 64)
    loss_diff = color_distribution_loss(x, y)
    assert loss_diff.item() > 0, "Different images loss should be > 0"


def test_color_distribution_loss_known_values():
    """Manual computation matches for known inputs."""
    from medsteer.losses import color_distribution_loss

    # Single image, 3 channels, 2x2
    gen = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        ]
    )
    tgt = torch.zeros_like(gen)

    # Manual: gen means = [2.5, 6.5, 10.5], gen stds = [std of 1,2,3,4, ...]
    # tgt means = [0,0,0], tgt stds = [0,0,0]
    gen_means = torch.tensor([2.5, 6.5, 10.5])
    gen_stds = gen[0].std(dim=[1, 2])
    loss_mean = torch.nn.functional.mse_loss(gen_means, torch.zeros(3))
    loss_std = torch.nn.functional.mse_loss(gen_stds, torch.zeros(3))
    expected = loss_mean + loss_std

    actual = color_distribution_loss(gen, tgt)
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Expected {expected.item()}, got {actual.item()}"
    )


def test_directions_save_load_roundtrip():
    from medsteer.directions import load_directions, save_directions

    # Build a small direction vectors dict
    dv = {}
    for s in range(3):
        dv[s] = {"blocks": [np.random.randn(1152).astype(np.float32) for _ in range(5)]}

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        tmp_path = f.name
    try:
        save_directions(dv, tmp_path)
        loaded = load_directions(tmp_path)
        for s in range(3):
            for b in range(5):
                assert np.allclose(dv[s]["blocks"][b], loaded[s]["blocks"][b]), (
                    f"Mismatch at step={s}, block={b}"
                )
    finally:
        os.unlink(tmp_path)


def test_directions_load_existing_pickle():
    """Load real steering vector file and verify structure."""
    if not os.path.exists(STEERING_VECTOR_FILE):
        raise RuntimeError(f"Steering vector file not found: {STEERING_VECTOR_FILE}")

    from medsteer.directions import load_directions

    dv = load_directions(STEERING_VECTOR_FILE)

    assert isinstance(dv, dict)
    # Should have 20 steps
    num_steps = len([k for k in dv if isinstance(k, int)])
    assert num_steps == 20, f"Expected 20 steps, got {num_steps}"

    # Each step should have 28 blocks
    for step in range(20):
        blocks = dv[step]["blocks"]
        assert len(blocks) == 28, f"Step {step}: expected 28 blocks, got {len(blocks)}"
        for b_idx, b in enumerate(blocks):
            arr = np.asarray(b)
            assert arr.shape == (1152,), (
                f"Step {step}, block {b_idx}: expected (1152,), got {arr.shape}"
            )
            # Check L2-normalized (norm ≈ 1)
            norm = np.linalg.norm(arr)
            assert abs(norm - 1.0) < 0.01, (
                f"Step {step}, block {b_idx}: norm={norm}, expected ~1.0"
            )


def test_default_output_filename():
    from medsteer.directions import default_output_filename

    result = default_output_filename(
        "PixArt-alpha/PixArt-XL-2-512x512",
        "dyed lifted polyps",
        "normal cecum",
    )
    assert result == "PixArt-XL-2-512x512_dyed lifted polyps_normal cecum.pickle", (
        f"Unexpected filename: {result}"
    )


def test_extract_seed():
    from medsteer.visualization.comparison_grid import extract_seed

    assert extract_seed("orig_seed42.png") == 42
    assert extract_seed("steered_seed7.png") == 7
    assert extract_seed("no_seed_here.png") is None


def test_abs_diff_heatmap():
    from PIL import Image

    from medsteer.visualization.comparison_grid import abs_diff_heatmap

    # Identical images → uniform heatmap (low diff)
    a = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    result = abs_diff_heatmap(a, a)
    arr = np.array(result)
    # Diff is 0 everywhere, so gray=0, the jet colormap at 0 should be uniform
    assert arr.shape == (64, 64, 3)
    # All pixels same
    assert np.all(arr == arr[0, 0]), "Identical images should produce uniform heatmap"

    # Different images → non-zero diff
    b = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    result2 = abs_diff_heatmap(a, b)
    arr2 = np.array(result2)
    assert np.any(arr2 != arr), "Different images should produce different heatmap"


def test_frechet_distance_identical():
    from medsteer.evaluation.fid import frechet_distance

    mu = np.zeros(10)
    sigma = np.eye(10)
    fid = frechet_distance(mu, sigma, mu, sigma)
    assert abs(fid) < 1e-6, f"Identical distributions FID should be ~0, got {fid}"


def test_frechet_distance_different():
    from medsteer.evaluation.fid import frechet_distance

    d = 10
    mu1 = np.zeros(d)
    mu2 = np.full(d, 8.0)  # shift by 8 in each dim
    sigma = np.eye(d)
    # FID = ||mu1 - mu2||^2 + trace(sigma1 + sigma2 - 2*sqrtm(sigma1 @ sigma2))
    # = 8^2 * 10 + trace(I + I - 2*I) = 640 + 0 = 640
    fid = frechet_distance(mu1, sigma, mu2, sigma)
    assert abs(fid - 640.0) < 1e-3, f"Expected FID=640, got {fid}"


def test_feature_stats():
    from medsteer.evaluation.fid import FeatureStats

    # 5 samples of 3 features
    data = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float64
    )
    stats = FeatureStats(dim=3)
    stats.update(data)
    mean, cov = stats.finalize()

    np_mean = data.mean(axis=0)
    np_cov = np.cov(data.T)  # ddof=1 by default
    assert np.allclose(mean, np_mean, atol=1e-10), f"Mean mismatch: {mean} vs {np_mean}"
    assert np.allclose(cov, np_cov, atol=1e-10), f"Cov mismatch:\n{cov}\nvs\n{np_cov}"


def test_kvasir_labels_ordering():
    from medsteer.classifier.dataset import KVASIR_LABELS

    expected = [
        "dyed lifted polyps",
        "dyed resection margins",
        "esophagitis",
        "normal cecum",
        "normal pylorus",
        "normal z-line",
        "polyps",
        "ulcerative colitis",
    ]
    assert KVASIR_LABELS == expected, f"Label mismatch: {KVASIR_LABELS}"


def test_make_grid_basic():
    from medsteer.visualization.comparison_grid import make_grid

    # 2 seeds × 3 columns, thumb=64
    seeds = [0, 1]
    baseline_imgs = {}  # no actual images — will draw gray boxes
    strength_imgs = {}
    columns = ["baseline", "s=1", "delta_s=1"]
    grid = make_grid(seeds, baseline_imgs, strength_imgs, columns, thumb=64, label_h=32)
    # Expected dimensions:
    # width = label_w(100) + 3 * 64 = 292
    # height = header_h(32) + 2 * (64 + 32) = 32 + 192 = 224
    assert grid.size == (292, 224), f"Unexpected grid size: {grid.size}"


# ===================================================================
# Level 3: Integration (GPU required, --gpu flag)
# ===================================================================


def test_attach_hooks_on_real_model():
    from diffusers import Transformer2DModel
    from medsteer import AttentionModulator, attach_hooks

    transformer = Transformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    mod = AttentionModulator(mode="passthrough")
    attach_hooks(transformer, mod)
    assert mod._total_blocks == 28, f"Expected 28 blocks, got {mod._total_blocks}"


def test_pipeline_baseline_generation():
    from medsteer import MedSteerPipeline

    pipeline = MedSteerPipeline.from_pretrained(
        MODEL_ID,
        lora_path=LORA_PATH,
        device="cuda",
    )
    image = pipeline.generate(
        "An endoscopic image of polyps",
        seed=42,
        num_steps=5,
        mode="baseline",
    )
    assert image.size == (512, 512), f"Expected 512x512, got {image.size}"


def test_pipeline_suppress_generation():
    from medsteer import MedSteerPipeline
    from medsteer.directions import load_directions

    pipeline = MedSteerPipeline.from_pretrained(
        MODEL_ID,
        lora_path=LORA_PATH,
        device="cuda",
    )
    dv = load_directions(STEERING_VECTOR_FILE)
    image = pipeline.generate(
        "An endoscopic image of dyed lifted polyps",
        seed=42,
        num_steps=5,
        mode="suppress",
        direction_vectors=dv,
        suppress_scale=2.0,
    )
    assert image.size == (512, 512), f"Expected 512x512, got {image.size}"


def test_pipeline_direction_vectors_path():
    """String path instead of pre-loaded dict should work."""
    from medsteer import MedSteerPipeline

    pipeline = MedSteerPipeline.from_pretrained(
        MODEL_ID,
        lora_path=LORA_PATH,
        device="cuda",
    )
    image = pipeline.generate(
        "An endoscopic image of dyed lifted polyps",
        seed=42,
        num_steps=5,
        mode="suppress",
        direction_vectors_path=STEERING_VECTOR_FILE,
        suppress_scale=2.0,
    )
    assert image.size == (512, 512), f"Expected 512x512, got {image.size}"


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Test medsteer package")
    parser.add_argument(
        "--gpu", action="store_true", help="Run GPU integration tests (Level 3)"
    )
    args = parser.parse_args()

    # Ensure github_pub/ is on sys.path so `import medsteer` works
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    print("=" * 60)
    print("MedSteer Test Suite")
    print("=" * 60)

    # --- Level 1: Import & Structure ---
    print("\n--- Level 1: Import & Structure ---")
    run_test("top_level_imports", test_top_level_imports)
    run_test("classifier_submodule_imports", test_classifier_submodule_imports)
    run_test("evaluation_submodule_imports", test_evaluation_submodule_imports)
    run_test("visualization_submodule_imports", test_visualization_submodule_imports)
    run_test("script_imports", test_script_imports)
    run_test("train_script_imports", test_train_script_imports)
    run_test("no_old_names_leak", test_no_old_names_leak)

    # --- Level 2: Unit Logic ---
    print("\n--- Level 2: Unit Logic ---")
    run_test("modulator_passthrough", test_modulator_passthrough)
    run_test(
        "modulator_passthrough_records_cache", test_modulator_passthrough_records_cache
    )
    run_test("modulator_suppress", test_modulator_suppress)
    run_test("modulator_stepping_logic", test_modulator_stepping_logic)
    run_test("modulator_suppress_negative_dot", test_modulator_suppress_negative_dot)
    run_test("modulator_step_clamping", test_modulator_step_clamping)
    run_test("modulator_cfg_batch_capture", test_modulator_cfg_batch_capture)
    run_test("color_distribution_loss", test_color_distribution_loss)
    run_test(
        "color_distribution_loss_known_values",
        test_color_distribution_loss_known_values,
    )
    run_test("directions_save_load_roundtrip", test_directions_save_load_roundtrip)
    run_test(
        "directions_load_existing_pickle",
        test_directions_load_existing_pickle,
        skip_if=not os.path.exists(STEERING_VECTOR_FILE),
        skip_reason=f"File not found: {STEERING_VECTOR_FILE}",
    )
    run_test("default_output_filename", test_default_output_filename)
    run_test("extract_seed", test_extract_seed)
    run_test("abs_diff_heatmap", test_abs_diff_heatmap)
    run_test("frechet_distance_identical", test_frechet_distance_identical)
    run_test("frechet_distance_different", test_frechet_distance_different)
    run_test("feature_stats", test_feature_stats)
    run_test("kvasir_labels_ordering", test_kvasir_labels_ordering)
    run_test("make_grid_basic", test_make_grid_basic)

    # --- Level 3: Integration (GPU) ---
    gpu_available = torch.cuda.is_available()
    print("\n--- Level 3: Integration (GPU) ---")
    run_test(
        "attach_hooks_on_real_model",
        test_attach_hooks_on_real_model,
        skip_if=not args.gpu,
        skip_reason="--gpu not set",
    )
    run_test(
        "pipeline_baseline_generation",
        test_pipeline_baseline_generation,
        skip_if=not args.gpu or not gpu_available,
        skip_reason="--gpu not set or no GPU" if not args.gpu else "no GPU available",
    )
    run_test(
        "pipeline_suppress_generation",
        test_pipeline_suppress_generation,
        skip_if=not args.gpu or not gpu_available,
        skip_reason="--gpu not set or no GPU" if not args.gpu else "no GPU available",
    )
    run_test(
        "pipeline_direction_vectors_path",
        test_pipeline_direction_vectors_path,
        skip_if=not args.gpu or not gpu_available,
        skip_reason="--gpu not set or no GPU" if not args.gpu else "no GPU available",
    )

    # --- Summary ---
    total = _results["pass"] + _results["fail"] + _results["skip"]
    print(f"\n{'=' * 60}")
    print(
        f"Results: {_results['pass']} passed, {_results['fail']} failed, {_results['skip']} skipped / {total} total"
    )
    print(f"{'=' * 60}")

    sys.exit(1 if _results["fail"] > 0 else 0)


if __name__ == "__main__":
    main()
