# embeddings/text_embedder.py

import os
import sys
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DIR

PREPROCESSED_CSV  = os.path.join(PROCESSED_DIR, "dataset_preprocessed.csv")
EMBEDDINGS_DIR    = os.path.join(PROCESSED_DIR, "embeddings")

# Output files
TITLE_EMB_FILE       = os.path.join(EMBEDDINGS_DIR, "title_embeddings.npy")
TRANSCRIPT_EMB_FILE  = os.path.join(EMBEDDINGS_DIR, "transcript_embeddings.npy")
THUMBTEXT_EMB_FILE   = os.path.join(EMBEDDINGS_DIR, "thumbnail_text_embeddings.npy")
COMMENTS_EMB_FILE    = os.path.join(EMBEDDINGS_DIR, "comments_embeddings.npy")
COMBINED_EMB_FILE    = os.path.join(EMBEDDINGS_DIR, "combined_embeddings.npy")
LABELS_FILE          = os.path.join(EMBEDDINGS_DIR, "labels.npy")
VIDEO_IDS_FILE       = os.path.join(EMBEDDINGS_DIR, "video_ids.npy")


def load_preprocessed():
    rows = []
    with open(PREPROCESSED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def batch_encode(model, texts, batch_size=64, desc=""):
    """
    Encode a list of texts in batches.
    Returns numpy array of shape (N, embedding_dim).
    """
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        # Replace empty strings with a placeholder
        batch = [t if t.strip() else "[empty]" for t in batch]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

        done = min(i + batch_size, total)
        print(f"    {desc}: {done}/{total} encoded...", end="\r")

    print(f"    {desc}: {total}/{total} encoded. ✓       ")
    return np.vstack(all_embeddings)


def generate_embeddings():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print("=" * 65)
    print("  PHASE 10 — Text Embedding Generation")
    print("  Model: all-MiniLM-L6-v2  (384-dim, fast, local)")
    print("=" * 65 + "\n")

    # ── Load data ─────────────────────────────────────────────────
    rows = load_preprocessed()
    total = len(rows)
    print(f"  Loaded {total} rows.\n")

    # ── Extract text columns ──────────────────────────────────────
    video_ids      = [r["video_id"]              for r in rows]
    titles         = [r["title_clean"]           for r in rows]
    transcripts    = [r["transcript_clean"]      for r in rows]
    thumbnail_texts= [r["thumbnail_text_clean"]  for r in rows]
    comments       = [r["comments_clean"]        for r in rows]
    combined       = [r["text_combined"]         for r in rows]
    labels         = [int(r["label"])            for r in rows]

    # ── Load model ────────────────────────────────────────────────
    print("  Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Model loaded. Embedding dim: 384\n")
    print("  Generating embeddings for each text field...\n")

    # ── Generate embeddings for each field ────────────────────────
    title_emb      = batch_encode(model, titles,          desc="Title         ")
    transcript_emb = batch_encode(model, transcripts,     desc="Transcript    ")
    thumbtext_emb  = batch_encode(model, thumbnail_texts, desc="Thumbnail text")
    comments_emb   = batch_encode(model, comments,        desc="Comments      ")
    combined_emb   = batch_encode(model, combined,        desc="Combined      ")

    labels_arr    = np.array(labels)
    video_ids_arr = np.array(video_ids)

    # ── Save all embeddings as .npy files ─────────────────────────
    np.save(TITLE_EMB_FILE,      title_emb)
    np.save(TRANSCRIPT_EMB_FILE, transcript_emb)
    np.save(THUMBTEXT_EMB_FILE,  thumbtext_emb)
    np.save(COMMENTS_EMB_FILE,   comments_emb)
    np.save(COMBINED_EMB_FILE,   combined_emb)
    np.save(LABELS_FILE,         labels_arr)
    np.save(VIDEO_IDS_FILE,      video_ids_arr)

    print(f"\n  [✓] All embeddings saved → {EMBEDDINGS_DIR}/")
    print(f"\n  Embedding shapes:")
    print(f"      Title:          {title_emb.shape}")
    print(f"      Transcript:     {transcript_emb.shape}")
    print(f"      Thumbnail text: {thumbtext_emb.shape}")
    print(f"      Comments:       {comments_emb.shape}")
    print(f"      Combined:       {combined_emb.shape}")
    print(f"      Labels:         {labels_arr.shape}")
    print(f"\n  Label distribution in embeddings:")
    for i, name in enumerate(["Not clickbait", "Good clickbait", "Bad clickbait"]):
        count = int(np.sum(labels_arr == i))
        print(f"      {i} - {name}: {count}")
    print(f"\n  [✓] Phase 10 complete. Ready for Phase 11 (image embeddings).\n")


if __name__ == "__main__":
    generate_embeddings()