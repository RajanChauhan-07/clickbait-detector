# collection/ocr_extractor.py

import os
import json
import sys
from PIL import Image
import pytesseract

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import THUMBNAILS_DIR, THUMBNAIL_TEXT_FILE, RAW_DIR


def extract_text_from_thumbnail(image_path):
    """
    Run OCR on a single thumbnail image.
    Returns extracted text as a cleaned string.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Config: treat image as a single block of text, optimized for bold overlay text
        custom_config = r"--oem 3 --psm 3"
        text = pytesseract.image_to_string(img, config=custom_config)

        # Clean up — strip extra whitespace and newlines
        cleaned = " ".join(text.split())
        return cleaned

    except Exception as e:
        return ""


def extract_all_thumbnail_text():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Load metadata to get video IDs in order
    metadata_path = os.path.join(RAW_DIR, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total   = len(metadata)
    success = 0
    empty   = 0
    failed  = 0

    print("=" * 60)
    print("  PHASE 4 — Thumbnail OCR Text Extraction")
    print(f"  Total videos: {total}")
    print("=" * 60)

    # Resume support
    if os.path.exists(THUMBNAIL_TEXT_FILE):
        with open(THUMBNAIL_TEXT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        already_done = {item["video_id"] for item in existing}
        results = existing
        print(f"\n  Resuming — {len(already_done)} already processed, skipping.\n")
    else:
        already_done = set()
        results = []

    for idx, item in enumerate(metadata, 1):
        video_id   = item["video_id"]
        image_path = os.path.join(THUMBNAILS_DIR, f"{video_id}.jpg")

        if video_id in already_done:
            continue

        if not os.path.exists(image_path):
            failed += 1
            results.append({"video_id": video_id, "thumbnail_text": ""})
            print(f"  [{idx}/{total}] {video_id}  →  ✗  image not found")
            continue

        text = extract_text_from_thumbnail(image_path)

        results.append({
            "video_id":       video_id,
            "thumbnail_text": text
        })

        if text:
            success += 1
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  [{idx}/{total}] {video_id}  →  ✓  \"{preview}\"")
        else:
            empty += 1
            print(f"  [{idx}/{total}] {video_id}  →  ~  (no text detected)")

        # Checkpoint every 100
        if idx % 100 == 0:
            with open(THUMBNAIL_TEXT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n  [checkpoint] {success} with text / {empty} empty / {failed} failed\n")

    # Final save
    with open(THUMBNAIL_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  [✓] Saved → {THUMBNAIL_TEXT_FILE}")
    print(f"  [✓] Phase 4 complete.")
    print(f"      Total:         {total}")
    print(f"      Text found:    {success}")
    print(f"      No text:       {empty}")
    print(f"      Image missing: {failed}\n")


if __name__ == "__main__":
    extract_all_thumbnail_text()