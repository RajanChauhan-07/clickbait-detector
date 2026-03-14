# model/fusion.py

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DIR

EMBEDDINGS_DIR = os.path.join(PROCESSED_DIR, "embeddings")

# Input embedding files
TITLE_EMB_FILE      = os.path.join(EMBEDDINGS_DIR, "title_embeddings.npy")
TRANSCRIPT_EMB_FILE = os.path.join(EMBEDDINGS_DIR, "transcript_embeddings.npy")
THUMBTEXT_EMB_FILE  = os.path.join(EMBEDDINGS_DIR, "thumbnail_text_embeddings.npy")
COMMENTS_EMB_FILE   = os.path.join(EMBEDDINGS_DIR, "comments_embeddings.npy")
IMAGE_EMB_FILE      = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
LABELS_FILE         = os.path.join(EMBEDDINGS_DIR, "labels.npy")
VIDEO_IDS_FILE      = os.path.join(EMBEDDINGS_DIR, "video_ids.npy")

# Output fusion file
FUSED_EMB_FILE      = os.path.join(EMBEDDINGS_DIR, "fused_embeddings.npy")
SIMILARITY_FILE     = os.path.join(EMBEDDINGS_DIR, "title_transcript_similarity.npy")


def cosine_similarity_rowwise(a, b):
    """
    Compute cosine similarity between corresponding rows of two matrices.
    Returns array of shape (N,)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a_norm * b_norm, axis=1)


def fuse_embeddings():
    print("=" * 65)
    print("  PHASE 12 — Multimodal Fusion")
    print("=" * 65 + "\n")

    # ── Load all embeddings ───────────────────────────────────────
    print("  Loading embeddings...")
    title_emb      = np.load(TITLE_EMB_FILE)
    transcript_emb = np.load(TRANSCRIPT_EMB_FILE)
    thumbtext_emb  = np.load(THUMBTEXT_EMB_FILE)
    comments_emb   = np.load(COMMENTS_EMB_FILE)
    image_emb      = np.load(IMAGE_EMB_FILE)
    labels         = np.load(LABELS_FILE)
    video_ids      = np.load(VIDEO_IDS_FILE)

    print(f"  Title:          {title_emb.shape}")
    print(f"  Transcript:     {transcript_emb.shape}")
    print(f"  Thumbnail text: {thumbtext_emb.shape}")
    print(f"  Comments:       {comments_emb.shape}")
    print(f"  Image (CLIP):   {image_emb.shape}")

    # ── Phase 13 preview — Title vs Transcript similarity ─────────
    # Compute semantic similarity between title and transcript
    # This becomes a feature in the fused vector
    print(f"\n  Computing title vs transcript similarity (Phase 13 signal)...")
    title_transcript_sim = cosine_similarity_rowwise(title_emb, transcript_emb)
    title_transcript_sim = title_transcript_sim.reshape(-1, 1)  # (N, 1)
    np.save(SIMILARITY_FILE, title_transcript_sim)
    print(f"  Similarity range: [{title_transcript_sim.min():.3f}, {title_transcript_sim.max():.3f}]")
    print(f"  Mean similarity:   {title_transcript_sim.mean():.3f}")

    # ── Fusion — concatenate all modalities ───────────────────────
    # Final vector per video:
    # [title(384) | transcript(384) | thumb_text(384) | comments(384) | image(512) | sim(1)]
    # Total = 384 + 384 + 384 + 384 + 512 + 1 = 2049 dims
    print(f"\n  Fusing all modalities via concatenation...")

    fused = np.concatenate([
        title_emb,            # 384  — what the title claims
        transcript_emb,       # 384  — what the video actually says
        thumbtext_emb,        # 384  — text overlaid on thumbnail
        comments_emb,         # 384  — viewer reactions
        image_emb,            # 512  — visual features of thumbnail
        title_transcript_sim, # 1    — semantic match score
    ], axis=1)

    print(f"  Fused shape: {fused.shape}")
    print(f"  Breakdown:")
    print(f"      Title embedding:          384")
    print(f"      Transcript embedding:     384")
    print(f"      Thumbnail text embedding: 384")
    print(f"      Comments embedding:       384")
    print(f"      Image embedding (CLIP):   512")
    print(f"      Title/transcript sim:       1")
    print(f"      ─────────────────────────────")
    print(f"      Total:                   2049")

    # ── Save fused embeddings ─────────────────────────────────────
    np.save(FUSED_EMB_FILE, fused)

    print(f"\n  [✓] Saved → {FUSED_EMB_FILE}")
    print(f"  [✓] Labels shape: {labels.shape}")
    print(f"\n  Label distribution:")
    for i, name in enumerate(["Not clickbait", "Good clickbait", "Bad clickbait"]):
        count = int(np.sum(labels == i))
        print(f"      {i} - {name}: {count}")
    print(f"\n  [✓] Phase 12 complete. Ready for Phase 13 + 14 (classifier training).\n")

    return fused, labels, video_ids


if __name__ == "__main__":
    fuse_embeddings()