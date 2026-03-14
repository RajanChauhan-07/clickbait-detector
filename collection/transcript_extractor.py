# collection/transcript_extractor.py

import os
import json
import time
from youtube_transcript_api import YouTubeTranscriptApi

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import METADATA_FILE, TRANSCRIPTS_FILE, RAW_DIR


def get_transcript(video_id):
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)

        full_text = " ".join(
            seg.text.strip()
            for seg in fetched
            if seg.text.strip()
        )
        full_text = " ".join(full_text.split())

        return {
            "video_id":     video_id,
            "transcript":   full_text,
            "language":     "en",
            "is_generated": True,
            "word_count":   len(full_text.split())
        }

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "too many" in error_msg.lower():
            print(f"  [!] Rate limited — sleeping 60s...")
            time.sleep(60)
            return {"video_id": video_id, "transcript": None, "reason": "rate_limited"}
        return {"video_id": video_id, "transcript": None, "reason": error_msg[:120]}


def extract_transcripts():
    os.makedirs(RAW_DIR, exist_ok=True)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    video_ids = [item["video_id"] for item in metadata]
    total = len(video_ids)

    print("=" * 60)
    print("  PHASE 2 — Transcript Extraction")
    print(f"  Total videos to process: {total}")
    print("=" * 60)

    # Resume support
    if os.path.exists(TRANSCRIPTS_FILE):
        with open(TRANSCRIPTS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        already_done = {item["video_id"] for item in existing}
        results = existing
        print(f"\n  Resuming — {len(already_done)} already processed, skipping.\n")
    else:
        already_done = set()
        results = []

    success = sum(1 for r in results if r.get("transcript"))
    failed  = sum(1 for r in results if not r.get("transcript"))

    for idx, video_id in enumerate(video_ids, 1):
        if video_id in already_done:
            continue

        result = get_transcript(video_id)
        results.append(result)

        if result.get("transcript"):
            success += 1
            status = f"✓  {result['word_count']} words"
        else:
            failed += 1
            status = f"✗  {result.get('reason', 'unknown')}"

        print(f"  [{idx}/{total}] {video_id}  →  {status}")

        # Checkpoint every 50 videos
        if idx % 50 == 0:
            with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n  [checkpoint] {success} success / {failed} failed\n")

        time.sleep(0.4)

    # Final save
    with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  [✓] Saved → {TRANSCRIPTS_FILE}")
    print(f"  [✓] Phase 2 complete.")
    print(f"      Total:   {total}")
    print(f"      Success: {success}  ({success/total*100:.1f}%)")
    print(f"      Failed:  {failed}\n")

    return results


if __name__ == "__main__":
    extract_transcripts()