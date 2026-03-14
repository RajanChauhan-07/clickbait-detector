# embeddings/image_embedder.py

import os
import sys
import csv
import numpy as np
from PIL import Image
import torch
import clip

csv.field_size_limit(sys.maxsize)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DIR, THUMBNAILS_DIR

PREPROCESSED_CSV  = os.path.join(PROCESSED_DIR, "dataset_preprocessed.csv")
EMBEDDINGS_DIR    = os.path.join(PROCESSED_DIR, "embeddings")
IMAGE_EMB_FILE    = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
IMAGE_IDS_FILE    = os.path.join(EMBEDDINGS_DIR, "image_video_ids.npy")


def load_preprocessed():
    rows = []
    with open(PREPROCESSED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_image_embeddings():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print("=" * 65)
    print("  PHASE 11 — Image Embedding Generation (CLIP)")
    print("=" * 65 + "\n")

    # ── Load CLIP model ───────────────────────────────────────────
    print("  Loading CLIP model (ViT-B/32)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"  Model loaded. Device: {device}\n")

    # ── Load dataset ──────────────────────────────────────────────
    rows  = load_preprocessed()
    total = len(rows)
    print(f"  Total videos: {total}\n")

    embeddings = []
    video_ids  = []
    missing    = 0
    processed  = 0

    # ── Process thumbnails in batches ─────────────────────────────
    batch_size = 32
    batch_images   = []
    batch_ids      = []

    def encode_batch(imgs, ids):
        img_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize
        return feats.cpu().numpy(), ids

    for idx, row in enumerate(rows, 1):
        video_id   = row["video_id"]
        thumb_path = row.get("thumbnail_path", "")

        # Handle missing thumbnails — use zero vector as placeholder
        if not thumb_path or not os.path.exists(thumb_path):
            # Try constructing path from video_id directly
            thumb_path = os.path.join(THUMBNAILS_DIR, f"{video_id}.jpg")

        if not os.path.exists(thumb_path):
            missing += 1
            # Zero vector placeholder for missing thumbnails
            embeddings.append(np.zeros(512, dtype=np.float32))
            video_ids.append(video_id)
            print(f"  [{idx}/{total}] {video_id}  →  ✗  missing (zero vector)")
            continue

        try:
            img = Image.open(thumb_path).convert("RGB")
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
            batch_ids.append(video_id)

            # Encode when batch is full
            if len(batch_images) == batch_size:
                feats, ids = encode_batch(batch_images, batch_ids)
                embeddings.extend(feats)
                video_ids.extend(ids)
                processed += len(ids)
                print(f"  Encoded {processed}/{total} thumbnails...", end="\r")
                batch_images = []
                batch_ids    = []

        except Exception as e:
            missing += 1
            embeddings.append(np.zeros(512, dtype=np.float32))
            video_ids.append(video_id)
            print(f"  [{idx}/{total}] {video_id}  →  ✗  error: {str(e)[:60]}")

    # Encode remaining batch
    if batch_images:
        feats, ids = encode_batch(batch_images, batch_ids)
        embeddings.extend(feats)
        video_ids.extend(ids)
        processed += len(ids)

    # ── Save embeddings ───────────────────────────────────────────
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    video_ids_arr  = np.array(video_ids)

    np.save(IMAGE_EMB_FILE, embeddings_arr)
    np.save(IMAGE_IDS_FILE, video_ids_arr)

    print(f"\n\n  [✓] Saved → {IMAGE_EMB_FILE}")
    print(f"\n  Summary:")
    print(f"      Total:     {total}")
    print(f"      Encoded:   {processed}")
    print(f"      Missing:   {missing}  (zero vectors)")
    print(f"      Shape:     {embeddings_arr.shape}  (512-dim CLIP features)")
    print(f"\n  [✓] Phase 11 complete. Ready for Phase 12 (multimodal fusion).\n")


if __name__ == "__main__":
    generate_image_embeddings()