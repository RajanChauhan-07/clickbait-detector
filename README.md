---
title: YouTube Clickbait Detector
emoji: 🎯
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# 🎯 YouTube Clickbait Detector
### Multimodal AI System — Text + Vision + Semantics

> **Author:** Rajan Chauhan | Bennett University | B.Tech Computer Science  
> **Built:** March 2026 | **Accuracy:** 80.51% | **Dataset:** 1,177 YouTube Videos

---

## Overview

A full end-to-end multimodal machine learning system that detects whether a YouTube video is clickbait by analyzing its title, transcript, thumbnail image, thumbnail text (OCR), and viewer comments simultaneously.

The system classifies videos into three categories:

| Label | Description |
|-------|-------------|
| **Not Clickbait** | Title accurately and plainly describes the content |
| **Good Clickbait** | Title is dramatic but the video genuinely delivers |
| **Bad Clickbait** | Title promises something the video does not deliver |

---

## Architecture

```
YouTube URL
    │
    ├─── YouTube Data API v3 ──────────► Title, Description, Thumbnail URL
    ├─── YouTube Transcript API ────────► Full video transcript
    ├─── YouTube Comments API ──────────► Top 20 viewer comments
    └─── Thumbnail Download + OCR ──────► Embedded text in thumbnail
              │
              ▼
    ┌─────────────────────────────────────────────────────┐
    │                  PREPROCESSING                       │
    │  Lowercase · Remove URLs · Remove punctuation        │
    │  Remove noise · Clean per field separately           │
    └─────────────────────────────────────────────────────┘
              │
              ├─── Text Embeddings (MiniLM-L6-v2, 384-dim each)
              │       ├── Title embedding
              │       ├── Transcript embedding
              │       ├── Thumbnail text embedding
              │       └── Comments embedding
              │
              ├─── Image Embedding (CLIP ViT-B/32, 512-dim)
              │       └── Thumbnail visual features
              │
              └─── Claim Verification
                      └── Cosine similarity (title vs transcript) → 1-dim
              │
              ▼
    ┌─────────────────────────────────────────────────────┐
    │              MULTIMODAL FUSION                       │
    │  Concatenation: 384+384+384+384+512+1 = 2049 dims   │
    └─────────────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────┐
    │            MLP CLASSIFIER                            │
    │  2049 → 512 → 256 → 128 → 3 (Softmax)               │
    │  Class-weighted · Early stopping · Adam optimizer    │
    └─────────────────────────────────────────────────────┘
              │
              ▼
    Not Clickbait / Good Clickbait / Bad Clickbait
```

---

## Project Structure

```
clickbait-detector/
│
├── config/
│   └── config.py                  # API keys, paths, constants, search queries
│
├── data/
│   ├── raw/
│   │   ├── metadata.json          # Phase 1 — 1177 video metadata records
│   │   ├── transcripts.json       # Phase 2 — 825 transcripts (70.1%)
│   │   ├── comments.json          # Phase 5 — 1108 comment sets (94.1%)
│   │   └── thumbnail_text.json    # Phase 4 — 354 OCR extractions
│   ├── thumbnails/                # Phase 3 — 1176 downloaded JPG images
│   └── processed/
│       ├── dataset.csv            # Phase 6 — unified dataset
│       ├── dataset_labeled.csv    # Phase 7 — auto-labeled dataset
│       ├── dataset_preprocessed.csv # Phase 9 — cleaned dataset
│       ├── embeddings/            # Phase 10–11 — .npy embedding files
│       └── models/                # Phase 14 — trained classifier + scaler
│
├── collection/
│   ├── youtube_scraper.py         # Phase 1 — metadata via YouTube Data API v3
│   ├── transcript_extractor.py    # Phase 2 — transcripts via youtube-transcript-api
│   ├── thumbnail_downloader.py    # Phase 3 — thumbnail images via requests
│   ├── ocr_extractor.py           # Phase 4 — OCR via pytesseract + PIL
│   └── comment_collector.py       # Phase 5 — comments via YouTube Data API v3
│
├── dataset/
│   ├── build_dataset.py           # Phase 6 — merges all raw data into CSV
│   └── label_dataset.py           # Phase 7 — heuristic + similarity auto-labeling
│
├── preprocessing/
│   └── preprocess.py              # Phase 9 — text cleaning pipeline
│
├── embeddings/
│   ├── text_embedder.py           # Phase 10 — MiniLM-L6-v2 text embeddings
│   └── image_embedder.py          # Phase 11 — CLIP ViT-B/32 image embeddings
│
├── model/
│   ├── fusion.py                  # Phase 12 — concatenation fusion + similarity
│   ├── claim_verifier.py          # Phase 13 — title vs transcript analysis
│   ├── classifier.py              # Phase 14+15 — MLP training + evaluation
│   └── train.py                   # Full pipeline entry point
│
├── evaluation/
│   └── evaluate.py                # Phase 15 — metrics, confusion matrix, plots
│
├── app/
│   ├── app.py                     # Flask dashboard backend
│   ├── templates/index.html       # Apple-style UI
│   └── static/
│       ├── style.css              # Light mode design system
│       └── main.js                # Animated result rendering
│
├── requirements.txt
└── README.md
```

---

## The 15-Phase Pipeline

### Phase 1 — YouTube Metadata Collection
**File:** `collection/youtube_scraper.py`

Used the YouTube Data API v3 to search for videos using 25 carefully chosen queries spanning both clickbait-heavy terms ("you won't believe", "shocking", "EXPOSED") and non-clickbait terms ("how to tutorial", "documentary", "lecture university"). Extracted video ID, title, description, thumbnail URL, view count, like count, and tags for each video. Collected 1,177 unique videos after deduplication.

**Why:** A diverse query set ensures the dataset contains all three label classes rather than being dominated by one type of content.

### Phase 2 — Transcript Extraction
**File:** `collection/transcript_extractor.py`

Used `youtube-transcript-api` (v1.2.4) to fetch subtitles for each video. Combined all subtitle segments into a single clean transcript string. Implemented resume support via checkpointing every 50 videos — if the script crashes, it picks up from where it stopped.

**Result:** 825 transcripts successfully extracted (70.1%). The 30% failure rate is expected — many videos have transcripts disabled or unavailable.

**Why:** The transcript is the ground truth of what a video actually contains. It's the most powerful signal for detecting whether a title's promise matches the content.

### Phase 3 — Thumbnail Download
**File:** `collection/thumbnail_downloader.py`

Downloaded all thumbnail images using the URLs collected in Phase 1. Saved as `<video_id>.jpg` in `data/thumbnails/`. Used a persistent requests session with a User-Agent header for reliability.

**Result:** 1,176/1,177 thumbnails downloaded (99.9%).

**Why:** Thumbnails are a critical clickbait signal — exaggerated facial expressions, bold overlay text, dramatic color palettes, and red arrows are common clickbait patterns that a vision model can detect.

### Phase 4 — Thumbnail OCR Text Extraction
**File:** `collection/ocr_extractor.py`

Applied Optical Character Recognition (Tesseract via pytesseract) on each thumbnail image to extract any embedded text. Used `--psm 3` (fully automatic page segmentation) optimized for bold overlay text common in YouTube thumbnails.

**Result:** 354 thumbnails contained extractable text (30%). Examples: "SHOCKING", "$3000 HILUX", "WAKEUPLIKEA", "YOU WON'T BELIEVE THIS".

**Why:** Thumbnail text is an independent signal from the title — many clickbait videos use ALL CAPS text, exclamation marks, and sensational phrases overlaid on the thumbnail image.

### Phase 5 — Comment Collection
**File:** `collection/comment_collector.py`

Fetched the top 20 most relevant comments per video using the YouTube Comments API, ordered by relevance. Comments are a proxy for viewer sentiment — comments like "clickbait", "misleading", and "waste of time" are strong bad clickbait signals, while "actually delivered" and "underrated" indicate good content.

**Result:** 1,108/1,177 videos had comments collected (94.1%). 69 videos had comments disabled.

### Phase 6 — Dataset Construction
**File:** `dataset/build_dataset.py`

Merged all five raw data sources (metadata, transcripts, comments, thumbnail text, thumbnail paths) into a single unified CSV and JSON dataset. Each row represents one video with all collected features plus an empty label column.

**Result:** 1,177 rows. 787 videos are "fully complete" (have transcript + thumbnail + comments simultaneously).

### Phase 7 — Automated Data Labeling
**File:** `dataset/label_dataset.py`

Instead of manual labeling (impractical at 1,177 videos), implemented a two-layer automated labeling pipeline:

**Layer 1 — Rule-based Heuristics (814 videos)**

Fast pattern matching using keyword lists and regex:
- Videos with titles matching patterns like "how to", "tutorial", "review", "documentary" → `not_clickbait`
- Titles containing "you won't believe", "shocking", "i almost died", "exposed" → `bad_clickbait`
- Titles with >60% uppercase letters → examined for good/bad clickbait markers
- Titles containing "24 hours", "i tried", "vs", "challenge" → `good_clickbait`

**Layer 2 — Title vs Transcript Semantic Similarity (363 videos)**

For videos where heuristics were uncertain, computed cosine similarity between the title embedding and transcript embedding using `all-MiniLM-L6-v2`:
- Similarity ≥ 0.30 → title matches content → `not_clickbait` or `good_clickbait`
- Similarity < 0.15 → content diverges from title → `bad_clickbait`
- Mid-range → fallback to keyword signals

**Label Distribution:**

| Label | Count | Percentage |
|-------|-------|------------|
| Not Clickbait | 708 | 60.2% |
| Good Clickbait | 277 | 23.5% |
| Bad Clickbait | 192 | 16.3% |

**Why this approach:** Manual labeling of 1,177 videos is time-prohibitive and introduces human fatigue bias. The two-layer approach mirrors industry practice for large-scale dataset construction — heuristics handle obvious cases instantly, semantic similarity handles nuanced ones.

### Phase 8 — Dataset Size Planning

Target was 500–1,000 videos per class. With 708/277/192 distribution, all classes exceed the 150-video minimum for meaningful training. The imbalance (not_clickbait dominates) is handled at training time via class weighting.

### Phase 9 — Preprocessing
**File:** `preprocessing/preprocess.py`

Cleaned each text field separately with a tailored pipeline:
- Lowercase conversion
- URL removal
- HTML tag stripping
- Non-ASCII character removal
- Punctuation removal (preserving apostrophes for contractions)
- Extra whitespace normalization

Also created a `text_combined` field merging all signals into one string, and converted string labels to integers (0/1/2).

**Why separate cleaning per field:** The title needs lighter cleaning than the transcript (to preserve its original signals). Thumbnail text needs different handling since OCR introduces noise characters.

### Phase 10 — Text Embedding Generation
**File:** `embeddings/text_embedder.py`

Encoded all text fields into dense numerical vectors using `sentence-transformers/all-MiniLM-L6-v2`:
- Architecture: 6-layer MiniLM transformer
- Output dimension: 384 per embedding
- Encoding: Processed in batches of 64 for efficiency

Generated separate embeddings for: title, transcript, thumbnail text, comments, and combined text.

**Why MiniLM over BERT:** MiniLM-L6-v2 is 5x smaller than BERT-base with only ~2% quality loss on sentence similarity tasks. For 1,177 × 5 fields on a local Mac, inference speed matters significantly.

**Output shapes:**

| Field | Shape |
|-------|-------|
| Title | (1177, 384) |
| Transcript | (1177, 384) |
| Thumbnail text | (1177, 384) |
| Comments | (1177, 384) |
| Combined | (1177, 384) |

### Phase 11 — Image Embedding Generation
**File:** `embeddings/image_embedder.py`

Processed all 1,176 thumbnail images through OpenAI's CLIP (Contrastive Language-Image Pretraining) model using the ViT-B/32 (Vision Transformer) backbone. Embeddings were L2-normalized before saving.

- Model: `ViT-B/32`
- Output dimension: 512 per image
- Device: Apple MPS (Metal Performance Shaders) for GPU acceleration
- Missing thumbnails: replaced with zero vectors

**Why CLIP over ResNet:** CLIP was trained on 400M image-text pairs with contrastive learning, making it particularly good at understanding the kind of dramatic, text-heavy thumbnail compositions common in YouTube videos. Its visual features capture semantic content rather than just pixel statistics.

**Output shape:** (1177, 512)

### Phase 12 — Multimodal Fusion
**File:** `model/fusion.py`

Combined all embeddings into a single unified representation per video using feature concatenation:

```
title(384) + transcript(384) + thumbnail_text(384) + comments(384) + image(512) + similarity(1)
= 2049 dimensions
```

Also computed the title vs transcript cosine similarity score as a standalone feature — this is the claim verification signal that directly encodes whether the title promise matches the video content.

**Why concatenation over attention fusion:** For this dataset size (1,177 samples), simple concatenation is more reliable than cross-modal attention which requires significantly more data to learn meaningful cross-modal relationships.

**Output shape:** (1177, 2049)

### Phase 13 — Claim Verification Module
**File:** `model/claim_verifier.py`

Analyzed how well the title vs transcript semantic similarity score separates the three classes. Ran threshold analysis to understand at which similarity cutoffs bad clickbait is best detected:

- Low similarity → title doesn't match content → likely bad clickbait
- High similarity → title matches content → not clickbait or good clickbait
- The similarity score is included as a feature in the fused vector

This module adds interpretability — the similarity score explains *why* the model made a prediction.

### Phase 14 — Classifier Training
**File:** `model/classifier.py`

Trained a Multi-Layer Perceptron on the fused 2049-dimensional embeddings:

**Architecture:**
```
Input (2049) → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Output(3, Softmax)
```

**Training configuration:**
- Optimizer: Adam (lr=0.001, adaptive)
- Regularization: L2 (α=0.001)
- Batch size: 64
- Early stopping: patience=15 iterations
- Validation fraction: 10% of training set
- Class weights: balanced (to handle imbalanced distribution)
- Train/test split: 80/20 stratified

**Class weights (computed automatically):**
The `balanced` class weight mode assigns higher weights to minority classes (good_clickbait, bad_clickbait) so the model doesn't just predict "not_clickbait" for everything.

### Phase 15 — Model Evaluation
**File:** `evaluation/evaluate.py`

**Results on held-out test set (236 videos):**

| Metric | Score |
|--------|-------|
| Accuracy | **80.51%** |
| F1 (Macro) | **0.705** |
| F1 (Weighted) | **0.784** |
| Precision (Macro) | 0.770 |
| Recall (Macro) | 0.697 |

**Per-class F1:**

| Class | F1 | Precision | Recall |
|-------|----|-----------|--------|
| Not Clickbait | 0.860 | 0.82 | 0.91 |
| Good Clickbait | 0.847 | 0.81 | 0.89 |
| Bad Clickbait | 0.407 | 0.69 | 0.29 |

**Confusion Matrix:**

| | Predicted Not | Predicted Good | Predicted Bad |
|--|-------------|----------------|---------------|
| **Actual Not** | 129 | 8 | 5 |
| **Actual Good** | 6 | 50 | 0 |
| **Actual Bad** | 23 | 4 | 11 |

**Analysis:**

Not Clickbait and Good Clickbait are detected with excellent F1 scores (0.86 and 0.85 respectively). Bad Clickbait has a lower F1 of 0.407 — primarily due to class imbalance (only 192 samples vs 708 not clickbait). The model misclassifies 23 bad clickbait videos as not clickbait, which is the hardest confusion pair since both classes can have plain-sounding titles with dramatically different content alignment.

---

## Flask Dashboard

**File:** `app/app.py`

An Apple-style web dashboard that accepts any YouTube URL and runs the complete inference pipeline live:

1. Fetches video metadata, transcript, comments and thumbnail from YouTube APIs
2. Encodes all text fields with MiniLM and the thumbnail with CLIP
3. Fuses into a 2049-dim vector
4. Scales with the trained StandardScaler
5. Predicts with the trained MLP classifier
6. Returns verdict, confidence score, class probabilities, and similarity signal

**Run:**
```bash
cd ~/clickbait-detector && PYTHONPATH=. python app/app.py
# Open http://127.0.0.1:5050
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data collection | YouTube Data API v3 | Metadata + comments |
| Transcripts | youtube-transcript-api | Subtitle extraction |
| OCR | Tesseract + pytesseract | Thumbnail text |
| Text embeddings | MiniLM-L6-v2 (384-dim) | Semantic text encoding |
| Image embeddings | CLIP ViT-B/32 (512-dim) | Visual feature extraction |
| Fusion | NumPy concatenation | Multimodal representation |
| Classifier | Scikit-learn MLPClassifier | 3-class prediction |
| Dashboard | Flask + vanilla JS | Live inference UI |
| Labeling | Heuristics + cosine similarity | Automated annotation |

---

## Installation

```bash
# Clone
git clone https://github.com/RajanChauhan-07/clickbait-detector
cd clickbait-detector

# Create virtual environment (Python 3.11 required)
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install google-api-python-client youtube-transcript-api requests \
    pillow torch torchvision sentence-transformers scikit-learn \
    joblib numpy flask pytesseract
pip install git+https://github.com/openai/CLIP.git

# macOS only — install Tesseract OCR engine
brew install tesseract

# Configure API key
# Edit config/config.py and set YOUTUBE_API_KEY
```

---

## Running the Full Pipeline

```bash
# Phase 1 — Collect metadata
python collection/youtube_scraper.py

# Phase 2 — Extract transcripts
python collection/transcript_extractor.py

# Phase 3 — Download thumbnails
python collection/thumbnail_downloader.py

# Phase 4 — OCR thumbnail text
python collection/ocr_extractor.py

# Phase 5 — Collect comments
python collection/comment_collector.py

# Phase 6 — Build dataset
python dataset/build_dataset.py

# Phase 7 — Auto-label
python dataset/label_dataset.py

# Phase 9 — Preprocess
python preprocessing/preprocess.py

# Phase 10 — Text embeddings
python embeddings/text_embedder.py

# Phase 11 — Image embeddings
python embeddings/image_embedder.py

# Phase 12 — Fuse
python model/fusion.py

# Phase 13 + 14 + 15 — Train and evaluate
python model/claim_verifier.py
python model/classifier.py

# Dashboard
PYTHONPATH=. python app/app.py
```

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total videos collected | 1,177 |
| Search queries used | 25 |
| Transcripts available | 825 (70.1%) |
| Thumbnails downloaded | 1,176 (99.9%) |
| Thumbnails with OCR text | 354 (30.1%) |
| Comments collected | 1,108 (94.1%) |
| Fully complete entries | 787 |
| Labeled via heuristics | 814 |
| Labeled via similarity | 363 |
| Train set | 941 |
| Test set | 236 |

---

## Key Design Decisions

**Why three classes instead of two?**
Binary clickbait detection misses an important nuance — some videos use dramatic titles but genuinely deliver on their promise. Penalizing these equally with deceptive clickbait would be unfair and inaccurate. The three-class system allows the model to reward good content creators who use engaging titles responsibly.

**Why automated labeling?**
Manual labeling of 1,177 videos would take 6–10 hours with significant fatigue bias. The two-layer automated approach (heuristics + semantic similarity) mirrors industry practice and produces labels that are reasonably consistent and scalable.

**Why concatenation fusion?**
For this dataset size, simple concatenation is more reliable than cross-modal attention or transformer fusion. Attention mechanisms require significantly more training data to learn meaningful cross-modal alignments. Concatenation preserves all information from each modality and lets the MLP learn which signals matter most.

**Why MiniLM over larger models?**
Running inference on a local Mac requires a balance between model quality and speed. MiniLM-L6-v2 achieves ~98% of BERT-base quality on sentence similarity benchmarks at 5x the inference speed with 6x fewer parameters.

**Why handle class imbalance with weights instead of resampling?**
With only 192 bad clickbait samples, SMOTE oversampling would generate synthetic embeddings that may not represent real-world bad clickbait patterns faithfully. Class weighting adjusts the loss function directly without modifying the data distribution.

---

## Limitations and Future Work

**Current limitations:**
- Bad Clickbait F1 of 0.407 due to only 192 training samples in that class
- Labels are automatically generated — some may be incorrect, especially for edge cases
- Only English-language videos are analyzed
- Transcript availability (70.1%) means 30% of videos are classified without content analysis

**Potential improvements:**
- Collect 500+ additional bad clickbait examples to balance the dataset
- Use GPT-4 or Claude for higher-quality automated labeling
- Fine-tune a RoBERTa or DeBERTa model on the labeled data instead of using frozen embeddings
- Add cross-modal attention between text and image embeddings
- Deploy to AWS EC2 with Nginx + Gunicorn for production use
- Add a feedback mechanism in the dashboard for users to correct predictions

---

## Results Summary

```
╔══════════════════════════════════════════════════════════════╗
║           CLICKBAIT DETECTOR — FINAL RESULTS                 ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset:        1,177 YouTube videos                        ║
║  Test set:       236 videos (20% stratified split)           ║
║                                                              ║
║  Accuracy:       80.51%                                      ║
║  F1 Macro:       0.705                                       ║
║  F1 Weighted:    0.784                                       ║
║                                                              ║
║  Not Clickbait F1:   0.860  ████████████████████            ║
║  Good Clickbait F1:  0.847  ████████████████████            ║
║  Bad Clickbait F1:   0.407  █████████                        ║
║                                                              ║
║  Text model:    MiniLM-L6-v2 (384-dim)                       ║
║  Image model:   CLIP ViT-B/32 (512-dim)                      ║
║  Fusion:        Concatenation (2049-dim)                     ║
║  Classifier:    MLP 2049→512→256→128→3                       ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Repository

**GitHub:** [github.com/RajanChauhan-07/clickbait-detector](https://github.com/RajanChauhan-07/clickbait-detector)
**Hugging Face:** https://huggingface.co/spaces/rajanchauhan/clickbait-detector

**Contact:** helllorajanchauhan@gmail.com  
**LinkedIn/GitHub:** RajanChauhan-07


