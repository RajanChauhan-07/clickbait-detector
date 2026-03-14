# dataset/label_dataset.py

import os
import sys
import csv
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATASET_CSV, LABELED_CSV, PROCESSED_DIR

# ── Load sentence transformer model (runs locally, no API) ───────────────────
print("\n  Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded.\n")

# ── Similarity thresholds ─────────────────────────────────────────────────────
# If title vs transcript similarity is below this → content doesn't match → bad clickbait
LOW_SIMILARITY_THRESHOLD  = 0.15
# If similarity is above this → content matches → not clickbait or good clickbait
HIGH_SIMILARITY_THRESHOLD = 0.30

# ── Heuristic keywords ────────────────────────────────────────────────────────
BAD_CLICKBAIT_WORDS = [
    "you won't believe", "shocking", "gone wrong", "gone sexual",
    "exposed", "they don't want you", "i almost died", "unbelievable",
    "omg", "must watch", "watch before", "deleted", "never seen before",
    "world's most", "gone too far", "i can't believe", "crying",
    "what they", "secret they", "the truth about", "they lied",
    "nobody talks about", "this is why", "you need to see",
    "insane", "crazy", "viral", "emotional", "heartbreaking",
    "not clickbait", "i got scammed", "she left me", "he left me",
    "we broke up", "i quit", "i got fired", "they kicked me out",
    "storytime", "story time"
]

GOOD_CLICKBAIT_WORDS = [
    "24 hours", "i tried", "30 days", "i bought", "cheapest",
    "most expensive", "vs", "challenge", "experiment", "i tested",
    "day in my life", "what happened when", "reaction", "i survived",
    "eating only", "living on", "i spent", "we tried", "rating",
    "ranking", "trying", "testing", "for a week", "for a month",
    "last to", "first to", "extreme", "$1 vs $1000"
]

NOT_CLICKBAIT_PATTERNS = [
    r"^how to ",
    r"\btutorial\b",
    r"\breview\b",
    r"\bexplained\b",
    r"\bdocumentary\b",
    r"\blecture\b",
    r"history of",
    r"guide to",
    r"full course",
    r"step by step",
    r"\bbeginner\b",
    r"introduction to",
    r"learn ",
    r"\bcourse\b",
    r"\blessons?\b",
    r"news report",
    r"analysis of",
    r"interview with",
    r"highlights",
    r"official video",
    r"official trailer",
    r"full match",
    r"full episode",
    r"live stream",
    r"\bpodcast\b",
    r"full album",
    r"compilation",
    r"top \d+",
]


def get_similarity(title, transcript):
    """
    Compute cosine similarity between title embedding and
    transcript summary embedding. Returns float 0.0–1.0.
    """
    if not transcript or len(transcript.strip()) < 50:
        return None  # not enough transcript to judge

    transcript_snippet = transcript[:512]

    title_emb      = embedder.encode([title])
    transcript_emb = embedder.encode([transcript_snippet])
    sim = cosine_similarity(title_emb, transcript_emb)[0][0]
    return float(sim)


def heuristic_label(title, thumbnail_text, comments):
    """
    Layer 1 — pure rule-based classification.
    Returns label or None if uncertain.
    """
    title_lower   = title.lower()
    thumb_lower   = (thumbnail_text or "").lower()
    comment_lower = (comments or "").split(" ||| ")[0][:200].lower()
    combined      = title_lower + " " + thumb_lower

    # Strong not-clickbait patterns
    for pattern in NOT_CLICKBAIT_PATTERNS:
        if re.search(pattern, title_lower):
            return "not_clickbait"

    # Viewer comment signals
    if any(w in comment_lower for w in [
        "clickbait", "misleading", "not what i expected",
        "lied", "waste of time", "false advertising"
    ]):
        return "bad_clickbait"

    if any(w in comment_lower for w in [
        "underrated", "actually delivers", "worth watching",
        "not clickbait", "actually good"
    ]):
        return "good_clickbait"

    # Strong bad clickbait words
    for word in BAD_CLICKBAIT_WORDS:
        if word in combined:
            return None  # uncertain — send to similarity check

    # Excessive caps → likely clickbait but need similarity to decide
    letters    = [c for c in title if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    if caps_ratio > 0.6:
        return None  # uncertain — send to similarity check

    # Good clickbait markers with no bad signals
    for word in GOOD_CLICKBAIT_WORDS:
        if word in combined:
            return "good_clickbait"

    # Plain title, no strong signals
    return "not_clickbait"


def similarity_label(title, transcript, thumbnail_text):
    """
    Layer 2 — title vs transcript semantic similarity.
    Called only when heuristics are uncertain.
    """
    sim = get_similarity(title, transcript)

    title_lower = title.lower()
    thumb_lower = (thumbnail_text or "").lower()
    combined    = title_lower + " " + thumb_lower

    has_good_markers = any(w in combined for w in GOOD_CLICKBAIT_WORDS)
    has_bad_markers  = any(w in combined for w in BAD_CLICKBAIT_WORDS)

    if sim is None:
        # No transcript — fall back to keyword signals only
        if has_bad_markers:
            return "bad_clickbait", sim
        if has_good_markers:
            return "good_clickbait", sim
        return "not_clickbait", sim

    # High similarity — title matches content
    if sim >= HIGH_SIMILARITY_THRESHOLD:
        if has_good_markers:
            return "good_clickbait", sim   # hyped title but delivers
        return "not_clickbait", sim        # plain title and delivers

    # Low similarity — title doesn't match content → bad clickbait
    if sim < LOW_SIMILARITY_THRESHOLD:
        return "bad_clickbait", sim

    # Mid similarity — ambiguous zone
    if has_good_markers:
        return "good_clickbait", sim
    if has_bad_markers:
        return "bad_clickbait", sim
    return "not_clickbait", sim


def load_rows():
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_labeled(rows):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def auto_label():
    rows  = load_rows()
    total = len(rows)

    print("=" * 65)
    print("  PHASE 7 — Automated Labeling")
    print("  Layer 1: Heuristics  |  Layer 2: Title vs Transcript similarity")
    print(f"  Total videos: {total}")
    print("=" * 65 + "\n")

    counts = {
        "not_clickbait":  0,
        "good_clickbait": 0,
        "bad_clickbait":  0,
    }
    method_counts = {"heuristic": 0, "similarity": 0}

    for idx, row in enumerate(rows, 1):
        title          = row.get("title", "")
        transcript     = row.get("transcript", "")
        thumbnail_text = row.get("thumbnail_text", "")
        comments       = row.get("comments", "")

        # Layer 1 — heuristics
        label  = heuristic_label(title, thumbnail_text, comments)
        method = "heuristic"
        sim    = None

        # Layer 2 — similarity if uncertain
        if label is None:
            label, sim = similarity_label(title, transcript, thumbnail_text)
            method = "similarity"

        row["label"] = label
        counts[label]        += 1
        method_counts[method] += 1

        sim_str = f"  sim={sim:.2f}" if sim is not None else ""
        print(f"  [{idx}/{total}] [{method:10}] {label:15}{sim_str}  |  {title[:50]}")

        # Checkpoint every 100
        if idx % 100 == 0:
            save_labeled(rows)
            print(f"\n  [checkpoint] Not: {counts['not_clickbait']} | "
                  f"Good: {counts['good_clickbait']} | "
                  f"Bad: {counts['bad_clickbait']}\n")

    save_labeled(rows)

    total_labeled = sum(counts.values())
    pct = lambda n: f"{n / total_labeled * 100:.1f}%"

    print(f"\n{'=' * 65}")
    print(f"  [✓] Labeling complete!")
    print(f"  [✓] Saved → {LABELED_CSV}")
    print(f"\n  Label distribution:")
    print(f"      Not clickbait:  {counts['not_clickbait']:4}  ({pct(counts['not_clickbait'])})")
    print(f"      Good clickbait: {counts['good_clickbait']:4}  ({pct(counts['good_clickbait'])})")
    print(f"      Bad clickbait:  {counts['bad_clickbait']:4}  ({pct(counts['bad_clickbait'])})")
    print(f"      Total:          {total_labeled}")
    print(f"\n  Method breakdown:")
    print(f"      Heuristic:      {method_counts['heuristic']}")
    print(f"      Similarity:     {method_counts['similarity']}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    auto_label()