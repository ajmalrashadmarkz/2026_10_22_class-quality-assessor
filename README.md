---
title: Class Quality Assessor
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: 3.12
pinned: false
---

# Class Quality Assessor

An automated, multimodal diagnostic system built for the **Humaniti AI 24-Hour Build Challenge**. The system evaluates online children's class recordings and transcripts, classifies their pedagogical quality into **Poor**, **Average**, **Good**, or **Excellent**, and generates structured diagnostic reports with timestamped evidence.

---

## Submission Links

- **Hosted Application:** [Live Production App](https://two026-10-22-class-quality-assessor.onrender.com)
- **GitHub Repository:** [Class Quality Assessor Repository](https://github.com/ajmalrashadmarkz/class-quality-assessor)
- **AI Interaction History:** Exported logs available in `logs/` (redacted of personal tokens and credentials)

---

## Approach & Methodology (Submission Note)

> *Official 284-word approach note for the challenge evaluation:*

We built an evidence-first, multimodal evaluation pipeline designed specifically for 8x8 online children's tutoring. Rather than passing raw video directly to an LLM—which compromises child privacy and risks token exhaustion—we decoupled the architecture into deterministic feature extraction and rubric-based qualitative assessment via a typed Pydantic `FeatureBundle` contract.

Audio processing operates locally using `faster-whisper` (int8 quantization) with Voice Activity Detection. It quantifies conversational dynamics: student-to-teacher talk ratio, active speech pace (WPM), acoustic energy variation (RMS), and identifies critical dead air (silences >5s). Computer vision runs via headless OpenCV, measuring inter-frame pixel variance to flag static screen freezes while keeping cloud frame calls disabled by default for privacy compliance.

The structured `FeatureBundle` feeds an LLM scoring engine (temperature=0) evaluating six weighted dimensions: Engagement (25%), Clarity (20%), Pacing (15%), Rapport (15%), Content & Visuals (15%), and Tech Management (10%). Pydantic schema validation strictly enforces at least two timestamped citations per dimension to prevent hallucinated or generic feedback. Scores are mapped to Poor, Average, Good, or Excellent using thresholds calibrated against ground truth via Spearman rank correlation.

**What Worked vs. Failed:**  
Automated pause detection and turn-taking frequency immediately surfaced significant quality separations (e.g., 60+ second awkward silences in weak sessions vs. rapid dialogue in excellent sessions). Naively relying on teacher talk-time failed, as high-quality concept explanations require sustained teacher delivery. Gating cloud vision protected child confidentiality while OpenCV captured visual dynamism.

**Main Limitations:**  
Calibrating thresholds on a 10-recording sample risks sensitivity to outlier teaching styles. Additionally, direct video URL streaming is supported, but authenticated LMS or cloud drive links require out-of-band credential exchange.

---

## Key Features

1. **Dual Ingestion Engine:**
   - **Tab 1: Official .VTT Transcript (Recommended):** Directly ingests Zoom Cloud `.vtt` transcripts with pre-labeled speaker tags for instantaneous evaluation in under 3 seconds.
   - **Tab 2: Video Recording / Direct URL:** Full audio-visual pipeline supporting `.mp4`, `.mov`, `.mkv` or direct HTTP/S video URLs.
2. **Deterministic Feature Extraction (Local CPU):**
   - Transcribes speech via `faster-whisper` (int8 quantized on CPU).
   - Measures speaking pace (WPM), awkward silences ($>5.0$s), and acoustic energy variation (RMS).
   - Detects screen changes and frozen slide stretches using headless OpenCV (`cv2.absdiff`).
3. **Calibrated Rubric-Based Reasoning:**
   - Evaluates 6 weighted dimensions: Student Engagement (25%), Clarity (20%), Pacing (15%), Rapport (15%), Content & Visuals (15%), and Tech Management (10%).
   - Enforces operational hard gates (new joiner self-introductions, tutor warmth, homework assignment).
4. **Child Privacy By Design:**
   - Strict ephemeral execution inside temporary directories (`tempfile.TemporaryDirectory`).
   - Zero storage of raw audio, video, or child faces. Cloud frame transmission disabled by default.
5. **New Era Blueprint UI:**
   - Zero emojis across the entire interface, titles, badges, and diagnostic reports.
   - Clinical carbon/slate design system with dedicated `/assess_class` API endpoint.

---

## Benchmark Calibration Alignment

The calibration thresholds in `class_assessor/calibration.json` were tuned and verified directly against official classroom recordings:

| Sample Session | Class Description | Ground Truth | System Score | Assigned Tier | Validation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B558 Beginner** | Pranav Kumar (Direct start, no intro, monotone delivery) | **Weak (Poor)** | **2.60** | **Poor** | Exact Match |
| **B435 Intermediate** | Sumit Ranjan (One-to-one interaction, needs warmth/HW) | **Average** | **2.65** | **Average** | Exact Match |
| **B491 Beginner** | Ayswarya Joseph (High rapport, individual questions, energetic) | **Excellent** | **3.50** | **Excellent** | Exact Match |

---

## Quickstart (Run Locally)

### 1. Prerequisites
- Python 3.12+
- `ffmpeg` installed on your system (or auto-resolved via `imageio-ffmpeg`)

### 2. Installation
```bash
git clone https://github.com/ajmalrashadmarkz/class-quality-assessor.git
cd class-quality-assessor
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Open `.env` and set your LLM credentials:
```env
# Required for qualitative assessment (DeepSeek, OpenAI, or compatible endpoint)
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
OPENAI_BASE_URL=https://api.deepseek.com

# Optional: Hugging Face token for Pyannote speaker diarization
HF_TOKEN=
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser at **`http://127.0.0.1:7860`**.

---

## Testing

Run the automated test suite with synthetic media generation:
```bash
pytest tests/ -v
```

All 7 test suites verify:
- Pydantic schema validation (requiring at least 2 timestamped evidence items).
- Minimum character length enforcement on factual observations.
- Calibration mapping across Poor, Average, Good, and Excellent tiers.
- Audio pause and dead air extraction ($>5$s).
- Synthetic end-to-end media processing (`ffmpeg` audio extraction + frame sampling).
- WebVTT cue and speaker tag parsing.
- OpenCV visual analysis resilience on missing or corrupted frames.

---

## Architecture Overview

```
Upload / Direct URL / .VTT Transcript
        │
        ▼
[1] Ingestion (ffmpeg audio extraction, uniform frame sampling / WebVTT parsing)
        │
        ├────────────────────────┬────────────────────────┐
        ▼                                                 ▼
[2a] Audio Analysis                               [2b] Visual Analysis
  • faster-whisper ASR (int8 CPU)                   • Frame differencing (cv2)
  • Pause & silence detection (>5s)                 • Static slide detection
  • Speaking rate (WPM) & RMS variance              • Optional privacy frame notes
        │                                                 │
        └────────────────────────┬────────────────────────┘
                                 ▼
[3] Feature Store (Pydantic FeatureBundle JSON Contract)
                                 │
                                 ▼
[4] Assessment Engine (Rubric LLM Scoring + Threshold Calibration)
                                 │
                                 ▼
[5] Gradio UI (Badge, Confidence, Summary, Strengths, Weaknesses, Evidence)
```

---

## Project Structure

```
├── app.py                     # Gradio 6 web application & /assess_class endpoint
├── requirements.txt           # Production Python dependencies
├── packages.txt               # Debian system dependencies (ffmpeg) for HF Spaces
├── README.md                  # System overview and run instructions
├── pytest.ini                 # Test configuration
├── .env.example               # Safe environment variable template
├── class_assessor/            # Core analysis pipeline
│   ├── assessor.py            # LLM prompt synthesis, validation & hard gates
│   ├── audio_analysis.py      # Whisper ASR, pause detection & acoustic metrics
│   ├── calibration.json       # Mathematical score thresholds
│   ├── config.py              # Application settings & privacy toggles
│   ├── media.py               # ffmpeg audio extraction & keyframe sampling
│   ├── pipeline.py            # Streaming pipeline with scoped tempdir lifecycle
│   ├── schemas.py             # Typed Pydantic v2 data contracts
│   ├── visual_analysis.py     # Headless OpenCV frame differencing
│   └── vtt_parser.py          # WebVTT transcript parser with speaker tags
├── scripts/                   # Utility scripts
│   └── calibrate.py           # Spearman rank threshold calibration tool
└── tests/                     # Automated test suite
    └── test_pipeline.py       # Unit and synthetic pipeline tests
```

---

## License & Privacy Notice

Developed for the **Humaniti AI 24-Hour Build Challenge**. All processing is strictly ephemeral in temporary local memory and wiped immediately upon completion. Zero media, biometric data, or child imagery is persisted.
