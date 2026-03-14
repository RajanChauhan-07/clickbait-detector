# dataset/build_dataset.py

import os
import json
import csv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    METADATA_FILE, TRANSCRIPTS_FILE,
    COMMENTS_FILE, THUMBNAIL_TEXT_FILE,
    THUMBNAILS_DIR, DATASET_CSV, PROCESSED_DIR
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("  PHASE 6 — Dataset Construction")
    print("=" * 60)

    # ── Load all raw data ─────────────────────────────────────────
    print("\n  Loading raw data files...")
    metadata       = load_json(METADATA_FILE)
    transcripts    = load_json(TRANSCRIPTS_FILE)
    comments       = load_json(COMMENTS_FILE)
    thumbnail_text = load_json(THUMBNAIL_TEXT_FILE)

    # ── Index by video_id for O(1) lookup ─────────────────────────
    transcript_map = {
        r["video_id"]: r.get("transcript") or ""
        for r in transcripts
    }
    comments_map = {
        r["video_id"]: " ||| ".join(r.get("comments") or [])
        for r in comments
    }
    thumb_text_map = {
        r["video_id"]: r.get("thumbnail_text") or ""
        for r in thumbnail_text
    }

    print(f"  Metadata:        {len(metadata)} videos")
    print(f"  Transcripts:     {sum(1 for v in transcript_map.values() if v)} with text")
    print(f"  Comments:        {sum(1 for v in comments_map.values() if v)} with comments")
    print(f"  Thumbnail text:  {sum(1 for v in thumb_text_map.values() if v)} with OCR text")

    # ── Build unified dataset rows ────────────────────────────────
    print("\n  Assembling dataset...")
    rows = []

    for item in metadata:
        video_id = item["video_id"]

        thumbnail_path = os.path.join(THUMBNAILS_DIR, f"{video_id}.jpg")
        thumbnail_exists = os.path.exists(thumbnail_path)

        row = {
            "video_id":        video_id,
            "title":           item.get("title", ""),
            "description":     item.get("description", ""),
            "channel":         item.get("channel", ""),
            "published_at":    item.get("published_at", ""),
            "view_count":      item.get("view_count", "0"),
            "like_count":      item.get("like_count", "0"),
            "comment_count":   item.get("comment_count", "0"),
            "duration":        item.get("duration", ""),
            "tags":            "|".join(item.get("tags", [])),
            "thumbnail_url":   item.get("thumbnail_url", ""),
            "thumbnail_path":  thumbnail_path if thumbnail_exists else "",
            "transcript":      transcript_map.get(video_id, ""),
            "thumbnail_text":  thumb_text_map.get(video_id, ""),
            "comments":        comments_map.get(video_id, ""),
            "label":           ""   # to be filled in Phase 7
        }
        rows.append(row)

    # ── Save as CSV ───────────────────────────────────────────────
    fieldnames = [
        "video_id", "title", "description", "channel",
        "published_at", "view_count", "like_count", "comment_count",
        "duration", "tags", "thumbnail_url", "thumbnail_path",
        "transcript", "thumbnail_text", "comments", "label"
    ]

    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Save as JSON too ──────────────────────────────────────────
    dataset_json = DATASET_CSV.replace(".csv", ".json")
    with open(dataset_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # ── Summary stats ─────────────────────────────────────────────
    has_transcript    = sum(1 for r in rows if r["transcript"])
    has_thumb_text    = sum(1 for r in rows if r["thumbnail_text"])
    has_comments      = sum(1 for r in rows if r["comments"])
    has_thumbnail     = sum(1 for r in rows if r["thumbnail_path"])
    fully_complete    = sum(
        1 for r in rows
        if r["transcript"] and r["thumbnail_path"] and r["comments"]
    )

    print(f"\n  [✓] Saved → {DATASET_CSV}")
    print(f"  [✓] Saved → {dataset_json}")
    print(f"\n  Dataset summary:")
    print(f"      Total rows:          {len(rows)}")
    print(f"      Has transcript:      {has_transcript}")
    print(f"      Has thumbnail:       {has_thumbnail}")
    print(f"      Has thumbnail text:  {has_thumb_text}")
    print(f"      Has comments:        {has_comments}")
    print(f"      Fully complete:      {fully_complete}  (transcript + thumbnail + comments)")
    print(f"\n  [✓] Phase 6 complete. Ready for Phase 7 labeling.\n")

    return rows


if __name__ == "__main__":
    build_dataset()