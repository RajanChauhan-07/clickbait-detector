# config/config.py

import os

# ── API Keys ──────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
THUMBNAILS_DIR  = os.path.join(DATA_DIR, "thumbnails")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")

# ── Collection settings ───────────────────────────────────────────────────────
MAX_RESULTS_PER_QUERY = 50        # YouTube API max per request
MAX_VIDEOS_TOTAL      = 3000      # target dataset size
MAX_COMMENTS_PER_VIDEO = 20       # top comments per video

# ── Clickbait search queries ──────────────────────────────────────────────────
SEARCH_QUERIES = [
    "you won't believe this",
    "shocking truth about",
    "I almost died",
    "this changed everything",
    "they lied to us",
    "the real reason why",
    "what they don't want you to know",
    "I tried this for 30 days",
    "exposing the truth",
    "gone wrong",
    "gone sexual",
    "extreme challenge",
    "24 hours in",
    "I bought the cheapest",
    "vs the most expensive",
    "reacting to my old videos",
    "day in my life",
    "educational documentary",
    "how to tutorial",
    "science explained",
    "history of",
    "documentary full",
    "lecture university",
    "news report analysis",
    "product review honest",
]

# ── Output file names ─────────────────────────────────────────────────────────
METADATA_FILE       = os.path.join(RAW_DIR, "metadata.json")
TRANSCRIPTS_FILE    = os.path.join(RAW_DIR, "transcripts.json")
COMMENTS_FILE       = os.path.join(RAW_DIR, "comments.json")
THUMBNAIL_TEXT_FILE = os.path.join(RAW_DIR, "thumbnail_text.json")
DATASET_CSV         = os.path.join(PROCESSED_DIR, "dataset.csv")
LABELED_CSV         = os.path.join(PROCESSED_DIR, "dataset_labeled.csv")