from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image
import imagehash


class HashMethod(str, Enum):
    AHASH = "ahash"
    PHASH = "phash"
    DHASH = "dhash"
    WHASH = "whash"


@dataclass
class ImageHashConfig:
    method: HashMethod = HashMethod.PHASH
    hash_size: int = 16
    similarity_threshold: int = 10
    auto_crop_background: bool = True
    bg_tolerance: int = 25
    canonical_size: int = 256


# ==============================
#  LOW-LEVEL UTILS
# ==============================


def _load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGBA")


def _auto_crop_alpha(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def _auto_crop_bg_color(img: Image.Image, tolerance: int) -> Image.Image:
    img_rgb = img.convert("RGB")
    arr = np.asarray(img_rgb, dtype=np.int16)
    h, w, _ = arr.shape

    n = 5
    corners = np.concatenate(
        [
            arr[0:n, 0:n],
            arr[0:n, w - n : w],
            arr[h - n : h, 0:n],
            arr[h - n : h, w - n : w],
        ],
        axis=0,
    )
    bg_color = corners.mean(axis=(0, 1))


    dist = np.linalg.norm(arr - bg_color[None, None, :], axis=-1)
    mask = dist > tolerance

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img_rgb

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = img_rgb.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped


def _preprocess_image(img: Image.Image, cfg: ImageHashConfig) -> Image.Image:
    img = _auto_crop_alpha(img)

    if cfg.auto_crop_background:
        img = _auto_crop_bg_color(img, cfg.bg_tolerance)

    img = img.convert("RGB")
    max_side = cfg.canonical_size
    w, h = img.size
    scale = min(max_side / w, max_side / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (max_side, max_side), color=(0, 0, 0))
    offset_x = (max_side - new_w) // 2
    offset_y = (max_side - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    return canvas


def _compute_hash_from_image(img: Image.Image, cfg: ImageHashConfig):
    if cfg.method == HashMethod.AHASH:
        return imagehash.average_hash(img, hash_size=cfg.hash_size)
    elif cfg.method == HashMethod.PHASH:
        return imagehash.phash(img, hash_size=cfg.hash_size)
    elif cfg.method == HashMethod.DHASH:
        return imagehash.dhash(img, hash_size=cfg.hash_size)
    elif cfg.method == HashMethod.WHASH:
        return imagehash.whash(img, hash_size=cfg.hash_size)
    else:
        raise ValueError(f"Unsupported hash method: {cfg.method}")


def _hash_bits(cfg: ImageHashConfig) -> int:
    return cfg.hash_size * cfg.hash_size


def _distance_to_similarity_percent(distance: int, cfg: ImageHashConfig) -> float:
    total_bits = _hash_bits(cfg)
    distance = max(0, min(distance, total_bits))
    similarity = (1.0 - distance / total_bits) * 100.0
    return round(similarity, 2)


def _estimate_blur_score(img: Image.Image) -> float:
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )

    pad_h, pad_w = 1, 1
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    H, W = arr.shape
    lap = np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            region = padded[i : i + 3, j : j + 3]
            lap[i, j] = (region * kernel).sum()

    var = float(lap.var())
    return var


def _blur_label(score: float) -> str:
    if score < 200:
        return "very blurry"
    elif score < 500:
        return "blurry"
    elif score < 1200:
        return "normal"
    else:
        return "sharp"


def compute_image_hash(img_path: str | Path, cfg: ImageHashConfig):
    img = _load_image(img_path)
    img = _preprocess_image(img, cfg)
    return _compute_hash_from_image(img, cfg)


def _compare_images_detailed(
    img_path_1: str | Path, img_path_2: str | Path, cfg: ImageHashConfig
) -> Dict[str, Any]:
    img1 = _load_image(img_path_1)
    img2 = _load_image(img_path_2)

    prep1 = _preprocess_image(img1, cfg)
    prep2 = _preprocess_image(img2, cfg)

    hash1 = _compute_hash_from_image(prep1, cfg)
    hash2 = _compute_hash_from_image(prep2, cfg)

    distance = hash1 - hash2
    similarity = _distance_to_similarity_percent(distance, cfg)

    blur1 = _estimate_blur_score(prep1)
    blur2 = _estimate_blur_score(prep2)

    return {
        "hash1": hash1,
        "hash2": hash2,
        "distance": int(distance),
        "similarity_percent": similarity,
        "is_similar": distance <= cfg.similarity_threshold,
        "blur_score_1": blur1,
        "blur_score_2": blur2,
        "blur_label_1": _blur_label(blur1),
        "blur_label_2": _blur_label(blur2),
    }


def are_images_similar(
    img_path_1: str | Path, img_path_2: str | Path, cfg: ImageHashConfig
) -> Tuple[bool, int]:
    info = _compare_images_detailed(img_path_1, img_path_2, cfg)
    return info["is_similar"], info["distance"]


def similarity_percent(
    img_path_1: str | Path, img_path_2: str | Path, cfg: ImageHashConfig
) -> float:
    info = _compare_images_detailed(img_path_1, img_path_2, cfg)
    return info["similarity_percent"]


def analyze_pair(
    img_path_1: str | Path, img_path_2: str | Path, cfg: ImageHashConfig
) -> Dict[str, Any]:
    return _compare_images_detailed(img_path_1, img_path_2, cfg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageHash utility with auto-crop + blur detection"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="phash",
        choices=[m.value for m in HashMethod],
        help="Hash method (default: phash)",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="Hash size N -> N x N bits (default: 16)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Max Hamming distance to consider similar (default: 10)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable auto-crop background",
    )
    return parser.parse_args()


def _main_cli() -> None:
    args = _parse_args()

    cfg = ImageHashConfig(
        method=HashMethod(args.method),
        hash_size=args.hash_size,
        similarity_threshold=args.threshold,
        auto_crop_background=not args.no_crop,
    )

    img1 = input("Nhập đường dẫn ảnh 1: ").strip()
    img2 = input("Nhập đường dẫn ảnh 2: ").strip()

    info = analyze_pair(img1, img2, cfg)
    total_bits = _hash_bits(cfg)

    print("\n========== IMAGE HASH TEST ==========")
    print(f"Method          : {cfg.method.value}")
    print(f"Hash size       : {cfg.hash_size}x{cfg.hash_size} ({total_bits} bits)")
    print(f"Similarity thr. : distance <= {cfg.similarity_threshold}")
    print(f"Auto-crop background    : {cfg.auto_crop_background}")
    print("-------------------------------------")
    print(f"Image 1         : {img1}")
    print(f"  Blur score    : {info['blur_score_1']:.2f} ({info['blur_label_1']})")
    print(f"Image 2         : {img2}")
    print(f"  Blur score    : {info['blur_score_2']:.2f} ({info['blur_label_2']})")
    print("-------------------------------------")
    print(f"Hamming distance: {info['distance']}")
    print(f"Similarity      : {info['similarity_percent']:.2f}%")
    print(f"Is similar?     : {info['is_similar']}")
    print("=====================================\n")


if __name__ == "__main__":
    _main_cli()
