"""
Merge steered images into comparison grids.

Creates per-seed comparison strips showing baseline + multiple suppression
strength levels, with optional difference heatmaps.

Expected directory layout:
  {root}/baseline/orig_seed{N}.png
  {root}/suppress{S}/steered_seed{N}.png

Output: one PNG per seed group, saved to {output_dir}/seed{N}.png
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_seed(filename: str):
    """Extract the integer seed from filenames like orig_seed42.png or steered_seed42.png."""
    m = re.search(r"seed(\d+)", filename)
    return int(m.group(1)) if m else None


def collect_images(directory: Path) -> dict:
    """Return {seed: path} for all PNG images in a directory."""
    result = {}
    if not directory.is_dir():
        return result
    for p in sorted(directory.rglob("*.png")):
        seed = extract_seed(p.name)
        if seed is not None:
            result[seed] = p
    return result


def try_load_font(size: int):
    """Try to load a TTF font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def abs_diff_heatmap(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """Compute |img_a - img_b| and return a jet-colormapped heatmap."""
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(a - b)
    gray = np.mean(diff, axis=2)
    max_val = gray.max()
    if max_val > 0:
        gray = gray / max_val
    r = np.clip(1.5 - np.abs(gray * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(gray * 4 - 2), 0, 1)
    b_ch = np.clip(1.5 - np.abs(gray * 4 - 1), 0, 1)
    rgb = np.stack([r, g, b_ch], axis=2)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def make_grid(
    seeds: list,
    baseline_imgs: dict,
    strength_imgs: dict,
    columns: list,
    thumb: int = 256,
    label_h: int = 32,
) -> Image.Image:
    """
    Build a grid image: rows = seeds, columns = settings + diffs.

    Args:
        seeds: List of seed integers.
        baseline_imgs: {seed: Path} for baseline images.
        strength_imgs: {strength_value: {seed: Path}} for steered images.
        columns: List of column labels (e.g., ["baseline", "s=1", "delta_s=1", ...]).
        thumb: Thumbnail size in pixels.
        label_h: Height of label bars.

    Returns:
        PIL Image of the grid.
    """
    n_seeds = len(seeds)
    n_cols = len(columns)

    label_w = 100
    header_h = label_h

    grid_w = label_w + n_cols * thumb
    grid_h = header_h + n_seeds * (thumb + label_h)

    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = try_load_font(max(14, label_h // 2))
    small_font = try_load_font(max(10, label_h // 3))

    # Draw column headers
    draw.rectangle([0, 0, grid_w, header_h], fill=(40, 40, 40))
    for col_idx, col_name in enumerate(columns):
        x = label_w + col_idx * thumb
        fill = (255, 200, 80) if col_name.startswith("delta") else (255, 255, 255)
        draw.text((x + 4, 4), col_name, fill=fill, font=font)

    def load_img(path: Path):
        try:
            img = Image.open(path).convert("RGB")
            return img.resize((thumb, thumb), Image.LANCZOS)
        except Exception:
            return None

    for row_idx, seed in enumerate(seeds):
        y_label = header_h + row_idx * (thumb + label_h)
        y_img = y_label + label_h

        draw.rectangle([0, y_label, grid_w, y_label + label_h], fill=(50, 50, 50))
        draw.text((8, y_label + 4), f"seed {seed}", fill=(255, 255, 255), font=font)

        baseline_pil = None
        if seed in baseline_imgs:
            baseline_pil = load_img(baseline_imgs[seed])

        for col_idx, col_name in enumerate(columns):
            x = label_w + col_idx * thumb

            if col_name == "baseline":
                if baseline_pil is not None:
                    grid.paste(baseline_pil, (x, y_img))
                else:
                    draw.rectangle(
                        [x, y_img, x + thumb, y_img + thumb], fill=(60, 60, 60)
                    )

            elif col_name.startswith("delta_s="):
                strength_val = col_name.replace("delta_s=", "")
                steered_pil = None
                if strength_val in strength_imgs and seed in strength_imgs[strength_val]:
                    steered_pil = load_img(strength_imgs[strength_val][seed])
                if baseline_pil is not None and steered_pil is not None:
                    diff_img = abs_diff_heatmap(steered_pil, baseline_pil)
                    grid.paste(diff_img, (x, y_img))
                else:
                    draw.rectangle(
                        [x, y_img, x + thumb, y_img + thumb], fill=(60, 60, 60)
                    )

            elif col_name.startswith("s="):
                strength_val = col_name.replace("s=", "")
                if (
                    strength_val in strength_imgs
                    and seed in strength_imgs[strength_val]
                ):
                    steered_pil = load_img(strength_imgs[strength_val][seed])
                    if steered_pil is not None:
                        grid.paste(steered_pil, (x, y_img))
                    else:
                        draw.rectangle(
                            [x, y_img, x + thumb, y_img + thumb], fill=(80, 0, 0)
                        )
                        draw.text(
                            (x + 4, y_img + 4),
                            "ERR",
                            fill=(255, 0, 0),
                            font=small_font,
                        )
                else:
                    draw.rectangle(
                        [x, y_img, x + thumb, y_img + thumb], fill=(60, 60, 60)
                    )

    return grid


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge baseline + multi-strength steered images into per-seed comparison strips."
    )
    parser.add_argument("--root", type=str, required=True, help="Root image directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save merged images.")
    parser.add_argument("--thumb_size", type=int, default=256, help="Thumbnail size in pixels.")
    parser.add_argument("--strengths", type=str, default=None, help="Comma-separated strength values to include.")
    parser.add_argument("--label_height", type=int, default=32, help="Label bar height.")
    parser.add_argument("--seeds_per_image", type=int, default=1, help="Seeds per merged image.")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "merged_per_seed"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = root / "baseline"
    if not baseline_dir.is_dir():
        print(f"ERROR: {baseline_dir} not found.")
        sys.exit(1)

    baseline_imgs = collect_images(baseline_dir)

    # Discover strength directories (support both "beta{X}" and "suppress{X}" patterns)
    strength_dirs = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            m = re.match(r"(?:beta|suppress)([\d.]+)", child.name)
            if m:
                strength_val = m.group(1)
                if args.strengths and strength_val not in args.strengths.split(","):
                    continue
                strength_dirs[strength_val] = child

    if not strength_dirs:
        print("ERROR: No strength directories found under --root!")
        sys.exit(1)

    strength_imgs = {}
    for strength_val, strength_path in strength_dirs.items():
        strength_imgs[strength_val] = collect_images(strength_path)

    all_seeds = set(baseline_imgs.keys())
    for strength_val, imgs in strength_imgs.items():
        all_seeds.update(imgs.keys())
    all_seeds = sorted(all_seeds)

    strengths_sorted = sorted(strength_dirs.keys(), key=lambda s: float(s))
    columns = ["baseline"]
    for s in strengths_sorted:
        columns.append(f"s={s}")
        columns.append(f"delta_s={s}")

    print(f"Root:       {root}")
    print(f"Output:     {output_dir}")
    print(f"Strengths:  {', '.join(strengths_sorted)}")
    print(f"Seeds:      {len(all_seeds)} ({all_seeds[0]}..{all_seeds[-1]})")
    print(f"Columns:    {', '.join(columns)}")
    print(f"Seeds per image: {args.seeds_per_image}\n")

    chunks = []
    for i in range(0, len(all_seeds), args.seeds_per_image):
        chunks.append(all_seeds[i : i + args.seeds_per_image])

    for chunk_idx, seed_chunk in enumerate(chunks):
        seed_range = f"{seed_chunk[0]}-{seed_chunk[-1]}"
        print(f"  Generating grid for seeds {seed_range} ...")

        grid_img = make_grid(
            seeds=seed_chunk,
            baseline_imgs=baseline_imgs,
            strength_imgs=strength_imgs,
            columns=columns,
            thumb=args.thumb_size,
            label_h=args.label_height,
        )

        if len(chunks) == 1:
            out_name = "all_seeds.png"
        else:
            out_name = f"seeds_{seed_chunk[0]}-{seed_chunk[-1]}.png"

        out_path = output_dir / out_name
        grid_img.save(out_path, optimize=True)
        print(f"    -> {out_path}  ({grid_img.size[0]}x{grid_img.size[1]})")

    print(f"\nDone! {len(chunks)} grid(s) saved to {output_dir}")


if __name__ == "__main__":
    main()
