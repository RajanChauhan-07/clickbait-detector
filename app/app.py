# app/app.py

import os
import sys
import json
import re
import numpy as np
import joblib
import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import YOUTUBE_API_KEY, PROCESSED_DIR

app = Flask(__name__)

MODELS_DIR   = os.path.join(PROCESSED_DIR, "models")
LABEL_NAMES  = ["Not Clickbait", "Good Clickbait", "Bad Clickbait"]
LABEL_COLORS = ["#30d158", "#ffd60a", "#ff453a"]
LABEL_ICONS  = ["✅", "⚠️", "🚨"]
LABEL_DESC   = [
    "This video title accurately describes its content.",
    "The title is dramatic but the video delivers on its promise.",
    "The title is misleading — the content doesn't match the promise."
]

# ── Load models ───────────────────────────────────────────────────────────────
print("\n  [1/5] Loading SentenceTransformer...", flush=True)
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer("all-MiniLM-L6-v2")
print("  [1/5] Done.", flush=True)

print("  [2/5] Loading CLIP...", flush=True)
import torch
import clip
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print(f"  [2/5] Done. Device: {device}", flush=True)

print("  [3/5] Loading classifier and scaler...", flush=True)
scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.joblib"))
print("  [3/5] Done.", flush=True)

print("  [4/5] Building YouTube client...", flush=True)
from googleapiclient.discovery import build
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
print("  [4/5] Done.", flush=True)

print("  [5/5] Loading transcript API...", flush=True)
from youtube_transcript_api import YouTubeTranscriptApi
print("  [5/5] Done.", flush=True)

print("\n  ✓ All models loaded. Starting Flask...\n", flush=True)


def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.match(r"^[0-9A-Za-z_-]{11}$", url.strip()):
        return url.strip()
    return None


def fetch_metadata(video_id):
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()
    items = response.get("items", [])
    if not items:
        return None
    item     = items[0]
    snippet  = item["snippet"]
    stats    = item.get("statistics", {})
    thumbs   = snippet.get("thumbnails", {})
    thumb_url = (
        thumbs.get("maxres", {}).get("url") or
        thumbs.get("high",   {}).get("url") or
        thumbs.get("medium", {}).get("url") or ""
    )
    return {
        "title":         snippet.get("title", ""),
        "description":   snippet.get("description", "")[:500],
        "channel":       snippet.get("channelTitle", ""),
        "published_at":  snippet.get("publishedAt", "")[:10],
        "view_count":    int(stats.get("viewCount", 0)),
        "like_count":    int(stats.get("likeCount", 0)),
        "thumbnail_url": thumb_url,
    }


def fetch_transcript(video_id):
    try:
        ytt      = YouTubeTranscriptApi()
        segments = ytt.fetch(video_id)
        text     = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return " ".join(text.split())
    except Exception:
        return ""


def fetch_comments(video_id):
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=20,
            order="relevance",
            textFormat="plainText"
        ).execute()
        comments = []
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"].strip()
            if text:
                comments.append(text)
        return " ".join(comments[:5])
    except Exception:
        return ""


def fetch_thumbnail_image(url):
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def get_image_embedding(img):
    if img is None:
        return np.zeros(512, dtype=np.float32)
    try:
        tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()
    except Exception:
        return np.zeros(512, dtype=np.float32)


def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def build_feature_vector(title, transcript, description, comments, img):
    title_clean      = clean_text(title)
    transcript_clean = clean_text(transcript)
    comments_clean   = clean_text(comments)

    title_emb      = text_model.encode([title_clean])[0]
    transcript_emb = text_model.encode([transcript_clean if transcript_clean else "[empty]"])[0]
    thumbtext_emb  = text_model.encode(["[empty]"])[0]
    comments_emb   = text_model.encode([comments_clean if comments_clean else "[empty]"])[0]
    image_emb      = get_image_embedding(img)

    a   = title_emb / (np.linalg.norm(title_emb) + 1e-8)
    b   = transcript_emb / (np.linalg.norm(transcript_emb) + 1e-8)
    sim = np.array([float(np.dot(a, b))])

    # Must match training fusion exactly:
    # title(384) + transcript(384) + thumbtext(384) + comments(384) + image(512) + sim(1) = 2049
    fused = np.concatenate([
        title_emb,      # 384
        transcript_emb, # 384
        thumbtext_emb,  # 384
        comments_emb,   # 384
        image_emb,      # 512
        sim             # 1
    ]).reshape(1, -1)

    return fused, float(sim[0])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    url      = data.get("url", "").strip()
    video_id = extract_video_id(url)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL or video ID."}), 400

    try:
        meta = fetch_metadata(video_id)
        if not meta:
            return jsonify({"error": "Video not found or unavailable."}), 404

        transcript = fetch_transcript(video_id)
        comments   = fetch_comments(video_id)
        img        = fetch_thumbnail_image(meta["thumbnail_url"])

        fused, sim    = build_feature_vector(
            meta["title"], transcript,
            meta["description"], comments, img
        )
        fused_scaled  = scaler.transform(fused)
        prediction    = int(classifier.predict(fused_scaled)[0])
        probabilities = classifier.predict_proba(fused_scaled)[0].tolist()
        confidence    = float(max(probabilities)) * 100

        return jsonify({
            "video_id":       video_id,
            "title":          meta["title"],
            "channel":        meta["channel"],
            "published_at":   meta["published_at"],
            "view_count":     f"{meta['view_count']:,}",
            "like_count":     f"{meta['like_count']:,}",
            "thumbnail_url":  meta["thumbnail_url"],
            "transcript":     transcript[:300] + "..." if len(transcript) > 300 else transcript,
            "has_transcript": bool(transcript),
            "has_comments":   bool(comments),
            "prediction":     prediction,
            "label":          LABEL_NAMES[prediction],
            "color":          LABEL_COLORS[prediction],
            "icon":           LABEL_ICONS[prediction],
            "description":    LABEL_DESC[prediction],
            "confidence":     round(confidence, 1),
            "similarity":     round(sim * 100, 1),
            "probabilities": {
                LABEL_NAMES[i]: round(p * 100, 1)
                for i, p in enumerate(probabilities)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5050, host="0.0.0.0")