# collection/youtube_scraper.py

import os
import json
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    YOUTUBE_API_KEY, SEARCH_QUERIES,
    MAX_RESULTS_PER_QUERY, MAX_VIDEOS_TOTAL,
    METADATA_FILE, RAW_DIR
)


def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_videos(youtube, query, max_results=50):
    """Search YouTube and return a list of video IDs for a given query."""
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        try:
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=next_page_token,
                relevanceLanguage="en",
                safeSearch="none"
            )
            response = request.execute()

            for item in response.get("items", []):
                if item["id"]["kind"] == "youtube#video":
                    video_ids.append(item["id"]["videoId"])

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.5)  # be polite to the API

        except HttpError as e:
            print(f"  [!] HTTP error during search '{query}': {e}")
            break

    return video_ids


def get_video_metadata(youtube, video_ids):
    """
    Given a list of video IDs, fetch full metadata in batches of 50.
    Returns a list of metadata dicts.
    """
    metadata = []

    # YouTube API accepts up to 50 IDs per videos().list() call
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            request = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch)
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet    = item.get("snippet", {})
                statistics = item.get("statistics", {})
                thumbnails = snippet.get("thumbnails", {})

                # Pick highest-res thumbnail available
                thumb_url = (
                    thumbnails.get("maxres", {}).get("url") or
                    thumbnails.get("high",   {}).get("url") or
                    thumbnails.get("medium", {}).get("url") or
                    thumbnails.get("default",{}).get("url") or ""
                )

                metadata.append({
                    "video_id":       item["id"],
                    "title":          snippet.get("title", ""),
                    "description":    snippet.get("description", ""),
                    "channel":        snippet.get("channelTitle", ""),
                    "published_at":   snippet.get("publishedAt", ""),
                    "thumbnail_url":  thumb_url,
                    "view_count":     statistics.get("viewCount", "0"),
                    "like_count":     statistics.get("likeCount", "0"),
                    "comment_count":  statistics.get("commentCount", "0"),
                    "duration":       item.get("contentDetails", {}).get("duration", ""),
                    "tags":           snippet.get("tags", []),
                    "category_id":    snippet.get("categoryId", ""),
                })

            time.sleep(0.3)

        except HttpError as e:
            print(f"  [!] HTTP error fetching metadata batch {i}: {e}")

    return metadata


def collect_metadata():
    """
    Main function — runs all queries, deduplicates video IDs,
    fetches metadata, and saves to metadata.json.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    youtube = get_youtube_client()

    print("=" * 60)
    print("  PHASE 1 — YouTube Metadata Collection")
    print("=" * 60)

    # ── Step 1: Collect video IDs across all queries ──────────────
    all_video_ids = set()

    for idx, query in enumerate(SEARCH_QUERIES, 1):
        print(f"\n[{idx}/{len(SEARCH_QUERIES)}] Searching: '{query}'")
        ids = search_videos(youtube, query, max_results=MAX_RESULTS_PER_QUERY)
        new_ids = [vid for vid in ids if vid not in all_video_ids]
        all_video_ids.update(new_ids)
        print(f"  → Found {len(ids)} videos | {len(new_ids)} new | Total: {len(all_video_ids)}")

        if len(all_video_ids) >= MAX_VIDEOS_TOTAL:
            print(f"\n  [✓] Reached target of {MAX_VIDEOS_TOTAL} videos. Stopping search.")
            break

        time.sleep(1.0)  # avoid quota spikes between queries

    all_video_ids = list(all_video_ids)[:MAX_VIDEOS_TOTAL]
    print(f"\n  Total unique videos collected: {len(all_video_ids)}")

    # ── Step 2: Fetch full metadata for all IDs ───────────────────
    print("\n  Fetching full metadata (in batches of 50)...")
    metadata = get_video_metadata(youtube, all_video_ids)
    print(f"  → Metadata fetched for {len(metadata)} videos")

    # ── Step 3: Save to JSON ──────────────────────────────────────
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  [✓] Saved → {METADATA_FILE}")
    print(f"  [✓] Phase 1 complete. {len(metadata)} videos ready for Phase 2.\n")

    return metadata


if __name__ == "__main__":
    collect_metadata()