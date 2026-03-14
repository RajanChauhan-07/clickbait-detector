# preprocessing/preprocess.py

import os
import sys
import csv
import re
import string

csv.field_size_limit(sys.maxsize)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LABELED_CSV, PROCESSED_DIR

PREPROCESSED_CSV = os.path.join(PROCESSED_DIR, "dataset_preprocessed.csv")


def clean_text(text):
    """
    Full text cleaning pipeline:
    - lowercase
    - remove URLs
    - remove HTML tags
    - remove special characters
    - remove extra whitespace
    - remove punctuation
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove emojis and non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove punctuation except apostrophes (for contractions)
    text = re.sub(r"[^\w\s']", " ", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


def clean_title(text):
    """
    Title cleaning — lighter touch than transcript.
    Keep some structure, just normalize.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s']", " ", text)
    text = " ".join(text.split())

    return text.strip()


def clean_comments(comments_str):
    """
    Clean and combine top comments into one string.
    Comments are stored as 'comment1 ||| comment2 ||| ...'
    """
    if not comments_str:
        return ""

    comments = comments_str.split(" ||| ")
    cleaned  = [clean_text(c) for c in comments[:5]]  # top 5 comments only
    cleaned  = [c for c in cleaned if len(c) > 5]     # remove very short ones
    return " ".join(cleaned)


def clean_thumbnail_text(text):
    """
    Thumbnail text is usually short bold strings.
    Just normalize and clean.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())

    return text.strip()


def label_to_int(label):
    """Convert string label to integer class."""
    mapping = {
        "not_clickbait":  0,
        "good_clickbait": 1,
        "bad_clickbait":  2,
    }
    return mapping.get(label, -1)


def load_labeled_rows():
    rows = []
    with open(LABELED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def preprocess():
    print("=" * 65)
    print("  PHASE 9 — Data Preprocessing")
    print("=" * 65 + "\n")

    rows = load_labeled_rows()
    total = len(rows)
    print(f"  Loaded {total} rows from labeled dataset.\n")

    processed  = []
    skipped    = 0
    label_counts = {0: 0, 1: 0, 2: 0}

    for idx, row in enumerate(rows, 1):
        label_str = row.get("label", "").strip()
        label_int = label_to_int(label_str)

        # Skip rows with no valid label
        if label_int == -1:
            skipped += 1
            continue

        # ── Clean each text field separately ─────────────────────
        clean = {
            "video_id":            row.get("video_id", ""),
            "title_clean":         clean_title(row.get("title", "")),
            "description_clean":   clean_text(row.get("description", "")),
            "transcript_clean":    clean_text(row.get("transcript", "")),
            "thumbnail_text_clean":clean_thumbnail_text(row.get("thumbnail_text", "")),
            "comments_clean":      clean_comments(row.get("comments", "")),

            # Keep originals too — needed for embeddings later
            "title_raw":           row.get("title", ""),
            "transcript_raw":      row.get("transcript", "")[:1000],
            "thumbnail_path":      row.get("thumbnail_path", ""),

            # Label
            "label_str":           label_str,
            "label":               label_int,
        }

        # ── Combined text field — all signals merged ──────────────
        clean["text_combined"] = " ".join(filter(None, [
            clean["title_clean"],
            clean["thumbnail_text_clean"],
            clean["description_clean"][:200],
            clean["transcript_clean"][:300],
            clean["comments_clean"][:200],
        ]))

        processed.append(clean)
        label_counts[label_int] += 1

        if idx % 100 == 0:
            print(f"  Processed {idx}/{total}...")

    # ── Save preprocessed CSV ─────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    fieldnames = [
        "video_id",
        "title_clean", "description_clean", "transcript_clean",
        "thumbnail_text_clean", "comments_clean",
        "title_raw", "transcript_raw", "thumbnail_path",
        "text_combined", "label_str", "label"
    ]

    with open(PREPROCESSED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed)

    total_valid = len(processed)
    pct = lambda n: f"{n / total_valid * 100:.1f}%"

    print(f"\n  [✓] Saved → {PREPROCESSED_CSV}")
    print(f"\n  Summary:")
    print(f"      Total rows:      {total}")
    print(f"      Valid (labeled): {total_valid}")
    print(f"      Skipped:         {skipped}")
    print(f"\n  Class distribution:")
    print(f"      0 - Not clickbait:  {label_counts[0]:4}  ({pct(label_counts[0])})")
    print(f"      1 - Good clickbait: {label_counts[1]:4}  ({pct(label_counts[1])})")
    print(f"      2 - Bad clickbait:  {label_counts[2]:4}  ({pct(label_counts[2])})")
    print(f"\n  [✓] Phase 9 complete. Ready for Phase 10 (embeddings).\n")

    return processed


if __name__ == "__main__":
    preprocess()