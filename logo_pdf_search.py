import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def embed_images_clip(pil_images):
    inputs = clip_processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        return img_emb.cpu().numpy()


def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def sliding_windows(img_cv2, scales=(0.2, 0.3, 0.4), stride_ratio=0.35, min_size=60):
    h, w = img_cv2.shape[:2]
    windows = []
    for s in scales:
        win_h = max(int(h * s), min_size)
        win_w = max(int(w * s), min_size)
        stride_h = max(int(win_h * stride_ratio), 16)
        stride_w = max(int(win_w * stride_ratio), 16)
        for y in range(0, h - win_h + 1, stride_h):
            for x in range(0, w - win_w + 1, stride_w):
                windows.append((x, y, win_w, win_h))
    return windows


def visualize_matches(page_pil, matches, out_path):
    img = pil_to_cv2(page_pil)
    for m in matches:
        x, y, w, h = m["bbox"]
        score = m["score"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)


def search_logo_in_page(logo_pil, page_pil, local_thresh=0.90, max_windows=2000, top_per_page=5, debug=False):
    img_cv2 = pil_to_cv2(page_pil)
    windows = sliding_windows(img_cv2)

    if debug:
        print(f"[DEBUG] Num windows before limit: {len(windows)}")

    if len(windows) > max_windows:
        windows = windows[:max_windows]

    crops = []
    coords = []
    for (x, y, w, h) in windows:
        crop = img_cv2[y:y + h, x:x + w]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crops.append(crop_pil)
        coords.append((x, y, w, h))

    if not crops:
        return []

    batch_size = 32
    all_emb = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        emb = embed_images_clip(batch)
        all_emb.append(emb)
    all_emb = np.vstack(all_emb)

    logo_emb = embed_images_clip([logo_pil])[0]
    sims = (all_emb @ logo_emb).reshape(-1)

    if debug:
        print(f"[DEBUG] sim range: min={sims.min():.3f}, max={sims.max():.3f}")

    candidate_indices = [i for i, s in enumerate(sims) if s >= local_thresh]
    if not candidate_indices:
        return []

    candidate_indices = sorted(candidate_indices, key=lambda i: sims[i], reverse=True)
    candidate_indices = candidate_indices[:top_per_page]

    matches = []
    for idx in candidate_indices:
        score = float(sims[idx])
        x, y, w, h = coords[idx]
        matches.append({"bbox": [int(x), int(y), int(w), int(h)], "score": score})

    return matches


def render_pdf_to_images(pdf_path, dpi=200):
    return convert_from_path(pdf_path, dpi=dpi)


def process_single_pdf(pdf_path, logo_pil, out_dir, thresh):
    pdf_name = Path(pdf_path).name
    print(f"[INFO] Processing PDF: {pdf_name}")

    pages = render_pdf_to_images(pdf_path, dpi=200)

    results = []
    for page_idx, page_img in enumerate(tqdm(pages, desc=f"Pages {pdf_name}")):
        page_no = page_idx + 1
        matches = search_logo_in_page(logo_pil, page_img, local_thresh=thresh, max_windows=2000, top_per_page=5, debug=False)
        
        if matches:
            print(f"  -> Found {len(matches)} matches on page {page_no}")
            for m in matches:
                x, y, w, h = m["bbox"]
                print(f"     page={page_no}, bbox=({x}, {y}, {w}, {h}), similarity={m['score']:.4f}")

            results.append({"page": page_no, "matches": matches})
            debug_img_path = out_dir / f"{pdf_name}_page{page_no}_debug.jpg"
            visualize_matches(page_img, matches, str(debug_img_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="Search logo in PDF using CLIP embeddings")
    parser.add_argument("--logo", type=str, required=True, help="Path to logo image")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--pdf-dir", type=str, help="Path to directory containing PDF files")
    parser.add_argument("--out-dir", type=str, default="debug_output", help="Output directory for debug images")
    parser.add_argument("--thresh", type=float, default=0.90, help="Cosine similarity threshold (0-1)")

    args = parser.parse_args()

    logo_pil = Image.open(args.logo).convert("RGB")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = []
    if args.pdf:
        pdf_files.append(args.pdf)
    if args.pdf_dir:
        for f in os.listdir(args.pdf_dir):
            if f.lower().endswith(".pdf"):
                pdf_files.append(str(Path(args.pdf_dir) / f))

    if not pdf_files:
        print("[ERROR] No PDF provided. Use --pdf or --pdf-dir")
        return

    all_results = {}
    for pdf_path in pdf_files:
        res = process_single_pdf(pdf_path, logo_pil, out_dir, args.thresh)
        all_results[pdf_path] = res

    print("\n=== GLOBAL SUMMARY ===")
    found_any = False
    for pdf_path, res in all_results.items():
        pdf_name = Path(pdf_path).name
        if not res:
            print(f"{pdf_name}: NO MATCH")
        else:
            found_any = True
            print(f"{pdf_name}:")
            for page_info in res:
                page = page_info["page"]
                matches = page_info["matches"]
                best = max(matches, key=lambda m: m["score"])
                x, y, w, h = best["bbox"]
                print(f"  - Page {page}: best_similarity={best['score']:.4f}, bbox=({x}, {y}, {w}, {h})")

    if not found_any:
        print("Logo image NOT FOUND in any provided PDF.")


if __name__ == "__main__":
    main()
