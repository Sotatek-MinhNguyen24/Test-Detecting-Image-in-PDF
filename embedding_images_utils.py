from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import argparse
import os

from image_resize import load_image_with_auto_resize


MODEL_NAME = "clip-ViT-B-32"
model = SentenceTransformer(MODEL_NAME)


def embed_image(image_path: str) -> np.ndarray:
    img = load_image_with_auto_resize(image_path)
    emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=False)
    return emb


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    return float(np.dot(v1, v2))


def compare_images(path1: str, path2: str, threshold: float = 0.8):
    emb1 = embed_image(path1)
    emb2 = embed_image(path2)

    sim = cosine_similarity(emb1, emb2)

    size1_mb = os.path.getsize(path1) / (1024 * 1024)
    size2_mb = os.path.getsize(path2) / (1024 * 1024)

    print("========== IMAGE EMBEDDING SIMILARITY ==========")
    print(f"Model           : {MODEL_NAME}")
    print(f"Image 1         : {path1} ({size1_mb:.2f} MB)")
    print(f"Image 2         : {path2} ({size2_mb:.2f} MB)")
    print("-----------------------------------------------")
    print(f"Cosine similarity: {sim:.4f}")
    print(f"Similarity (%)   : {sim * 100:.2f}%")

    is_similar = sim >= threshold
    print(f"Threshold        : {threshold:.2f}")
    print(f"Is similar?      : {is_similar}")
    print("================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two images using CLIP embeddings")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold (default: 0.8)",
    )
    args = parser.parse_args()

    compare_images(args.image1, args.image2, threshold=args.threshold)
