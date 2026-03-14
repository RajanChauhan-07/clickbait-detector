# collection/comment_collector.py

import os
import json
import time
import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    YOUTUBE_API_KEY, METADATA_FILE,
    COMMENTS_FILE, RAW_DIR, MAX_COMMENTS_PER_VIDEO
)


def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def fetch_comments(youtube, video_id, max_comments=20):
    """
    Fetch top comments for a single video.
    Returns a list of comment strings.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            order="relevance",
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            text = (
                item["snippet"]["topLevelComment"]
                    ["snippet"]["textDisplay"]
                    .strip()
            )
            if text:
                comments.append(text)

    except HttpError as e:
        error_reason = str(e)
        if "commentsDisabled" in error_reason or "403" in error_reason:
            pass  # comments disabled — silent skip
        elif "404" in error_reason:
            pass  # video not found — silent skip
        else:
            print(f"  [!] HTTP error for {video_id}: {error_reason[:80]}")
    except Exception as e:
        print(f"  [!] Error for {video_id}: {str(e)[:80]}")

    return comments


def collect_all_comments():
    os.makedirs(RAW_DIR, exist_ok=True)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total = len(metadata)

    print("=" * 60)
    print("  PHASE 5 — Comment Collection")
    print(f"  Total videos: {total}")
    print(f"  Max comments per video: {MAX_COMMENTS_PER_VIDEO}")
    print("=" * 60)

    # Resume support
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        already_done = {item["video_id"] for item in existing}
        results = existing
        print(f"\n  Resuming — {len(already_done)} already processed, skipping.\n")
    else:
        already_done = set()
        results = []

    success  = sum(1 for r in results if r.get("comments"))
    disabled = sum(1 for r in results if not r.get("comments"))

    youtube = get_youtube_client()

    for idx, item in enumerate(metadata, 1):
        video_id = item["video_id"]

        if video_id in already_done:
            continue

        comments = fetch_comments(youtube, video_id, MAX_COMMENTS_PER_VIDEO)

        results.append({
            "video_id": video_id,
            "comments": comments,
            "count":    len(comments)
        })

        if comments:
            success += 1
            print(f"  [{idx}/{total}] {video_id}  →  ✓  {len(comments)} comments")
        else:
            disabled += 1
            print(f"  [{idx}/{total}] {video_id}  →  ~  (no comments)")

        # Checkpoint every 50
        if idx % 50 == 0:
            with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n  [checkpoint] {success} with comments / {disabled} empty\n")

        time.sleep(0.5)  # stay within quota

    # Final save
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  [✓] Saved → {COMMENTS_FILE}")
    print(f"  [✓] Phase 5 complete.")
    print(f"      Total:              {total}")
    print(f"      With comments:      {success}")
    print(f"      No/disabled:        {disabled}\n")


if __name__ == "__main__":
    collect_all_comments()