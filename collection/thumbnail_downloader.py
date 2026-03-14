# collection/thumbnail_downloader.py

import os
import json
import time
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import METADATA_FILE, THUMBNAILS_DIR


def download_thumbnail(video_id, url, session):
    """
    Download a single thumbnail image and save it as <video_id>.jpg.
    Returns True on success, False on failure.
    """
    save_path = os.path.join(THUMBNAILS_DIR, f"{video_id}.jpg")

    # Skip if already downloaded
    if os.path.exists(save_path):
        return True

    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception:
        return False


def download_all_thumbnails():
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total   = len(metadata)
    success = 0
    failed  = 0
    skipped = 0

    print("=" * 60)
    print("  PHASE 3 — Thumbnail Download")
    print(f"  Total videos: {total}")
    print("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for idx, item in enumerate(metadata, 1):
        video_id = item["video_id"]
        url      = item.get("thumbnail_url", "")
        save_path = os.path.join(THUMBNAILS_DIR, f"{video_id}.jpg")

        # Resume support — skip already downloaded
        if os.path.exists(save_path):
            skipped += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{total}] Skipping {video_id} (already exists)")
            continue

        if not url:
            failed += 1
            print(f"  [{idx}/{total}] {video_id}  →  ✗  no URL in metadata")
            continue

        ok = download_thumbnail(video_id, url, session)

        if ok:
            success += 1
            status = "✓"
        else:
            failed += 1
            status = "✗  download failed"

        print(f"  [{idx}/{total}] {video_id}  →  {status}")

        # Small delay to be polite
        time.sleep(0.1)

    print(f"\n  [✓] Phase 3 complete.")
    print(f"      Total:    {total}")
    print(f"      Downloaded: {success}")
    print(f"      Skipped:    {skipped}  (already existed)")
    print(f"      Failed:     {failed}")
    print(f"      Thumbnails saved → {THUMBNAILS_DIR}\n")


if __name__ == "__main__":
    download_all_thumbnails()