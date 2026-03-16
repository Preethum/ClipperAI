# 🎬 ClipperAI

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/Preethum/ClipperAI)

> **AI-powered video clipping pipeline** — Automatically extract, crop, subtitle, and render viral short-form clips from long-form video using local LLMs, computer vision, and a modular Plan-Then-Render architecture.

ClipperAI takes a long-form video (podcast, vlog, gaming stream, etc.), uses multi-agent LLM analysis to identify the highest-potential viral moments, and produces ready-to-upload vertical clips with smart cropping and animated subtitles — all running locally on your hardware.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Agent LLM Pipeline** | Scout model identifies candidates → Editor model curates final selection |
| 👁️ **Vision Analysis** | Keyframe-level visual context via multimodal VLMs (Qwen3-VL, etc.) |
| ✂️ **Smart Cropping** | YOLO-based person detection + scene-aware tracking for 9:16 reframing |
| 📝 **Animated Subtitles** | Word-level Whisper transcription → 14 built-in PyCaps templates |
| 🎬 **Single-Pass Rendering** | Plan-Then-Render architecture: plan once, render once per clip |
| ⚙️ **Scenario System** | Pre-built configs for TikTok and Podcast, fully customizable |
| 🔧 **Module Toggling** | Enable/disable any pipeline step (crop, subs, render) per scenario |
| 📋 **Manifest Checkpointing** | JSON manifest saved between phases — resume or inspect anytime |

---

## 🏗️ Architecture

ClipperAI uses a **Plan-Then-Render** pipeline. Each module enriches a shared JSON manifest, and the Renderer executes all transformations in a single pass per clip.

```
Source Video
     │
     ▼
┌─────────────────────────────┐
│  PHASE 1: PLAN              │
│                             │
│  ClipperM (required)        │  → Produces base manifest (timestamps, titles, scores)
│       ↓                     │
│  CropperM (optional)        │  → Enriches manifest with crop/scene data
│       ↓                     │
│  SubsM (optional)           │  → Enriches manifest with word-level transcription
│                             │
│  📋 clips_manifest.json     │  ← Checkpoint saved here
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  PHASE 2: RENDER            │
│                             │
│  RendererM (optional)       │  → Cuts, crops, burns subs in one pass per clip
│                             │
│  📁 Final .mp4 files        │
└─────────────────────────────┘
```

**Why this matters:** Traditional pipelines write intermediate video files at each step (3× disk I/O per clip). Plan-Then-Render reads the source once and writes the final output once — significantly faster and more resumable.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with CUDA 12.x (recommended for Whisper, YOLO, and vision models)
- **FFmpeg** (place binaries in `bin/` or have it on your system PATH)
- **LM Studio** running locally with loaded models (Scout + Editor + optional Vision)

### Installation

```bash
# Clone the repository
git clone https://github.com/Preethum/ClipperAI.git
cd ClipperAI

# Create and activate virtual environment
python -m venv clipper-venv
# Windows:
clipper-venv\Scripts\activate
# Linux/macOS:
source clipper-venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Pipeline

```bash
# TikTok scenario — optimized for short, punchy vertical clips
python src/scenarios/tiktok.py --input "your_video.mp4" --clips 5 --output "tiktok_output"

# Podcast scenario — longer, more conversational clips
python src/scenarios/podcast.py --input "podcast_episode.mp4" --clips 10 --output "podcast_output"

# Gaming scenario — AI-powered HUD detection, Audio Clumping and Cushioning
python src/scenarios/gaming_pipeline.py --input "gameplay.mp4" --model "Train/runs/detect/train/weights/best.pt" --output "gaming_output"
```

### What You'll See

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🚀  ClipperAI Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Source:  your_video.mp4
     Output:  tiktok_output
     Steps:   CLIPPER → CROPPER → SUBS → RENDERER

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎬  Step 1/4: CLIPPER — Finding clips
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Scout: deepseek-r1  |  Editor: gemma-3-27b
     ✅ 5 clips found (180s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✂️  Step 2/4: CROPPER — Analyzing scenes (9:16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     [1/5] $1,000 Per Push-Up!  (12 scenes: 4T/8L)
     [2/5] Ultimate Rock Paper Scissors  (8 scenes: 2T/6L)
     ...
     ✅ 5 clips enriched (35s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎬  Step 4/4: RENDERER — Exporting final clips
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     [1/5] 1_push_up_challenge.mp4 (65s) [crop+subs]
     ...
     ✅ 5 clips rendered (90s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎉  Pipeline Complete — 5 clips in 305s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📁 Project Structure

```
ClipperAI/
├── src/
│   ├── modules/                  # Core processing modules
│   │   ├── ClipperM.py           # AI clip extraction (Scout + Editor LLM pipeline)
│   │   ├── CropperM.py           # Smart cropping (YOLO + scene detection)
│   │   ├── SubsM.py              # Subtitle transcription (Whisper + PyCaps)
│   │   ├── RendererM.py          # Single-pass video compositor (Hardware Acceleration)
│   │   ├── AudioAnalyzerM.py     # Audio intensity & combat spike detector
│   │   ├── DetectionM.py         # YOLO & OCR scanning wrapper
│   │   ├── SafeEntryExit.py      # Cushioning fine-tuning via VLM/Audio
│   │   └── StitcherM.py          # Video stitching with progress bars
│   ├── scenarios/                # Pre-built pipeline configurations
│   │   ├── global_functions.py   # Pipeline orchestrator
│   │   ├── tiktok.py             # TikTok-optimized scenario
│   │   └── podcast.py            # Podcast-optimized scenario
│   └── utils/
│       └── path_utils.py         # Path resolution utilities
├── templates/                    # 14 subtitle style templates
├── bin/                          # FFmpeg binaries (optional)
├── requirements.txt
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## 🧩 Modules

### ClipperM — AI Clip Extraction

The intelligence layer. Analyzes the full video using a two-pass LLM pipeline to identify the best viral moments.

**How it works:**

1. **Transcription** — Whisper extracts the full transcript with word-level timestamps
2. **OCR Analysis** (optional) — EasyOCR detects on-screen text for scene change awareness
3. **Vision Analysis** (optional) — A multimodal VLM (e.g., Qwen3-VL) describes sampled keyframes
4. **Scout Pass** — A reasoning LLM (e.g., DeepSeek-R1) reads transcript chunks with context overlap and identifies candidate clips with virality metrics
5. **Editor Pass** — A second LLM curates the final selection, enforcing diversity across viral archetypes and rejecting incomplete stories
6. **Deduplication** — Overlapping clips from chunked processing are merged automatically

**Output:** A manifest of clip metadata — timestamps, titles, scores, anchors — no video files written.

---

### CropperM — Smart Vertical Cropping

Enriches the manifest with per-clip scene analysis and cropping strategies.

**How it works:**

1. **Scene Detection** — PySceneDetect identifies cuts/transitions across the full video (runs once)
2. **Person Detection** — YOLOv8 locates people in each scene segment
3. **Strategy Selection** — Per-scene: `TRACK` (follow a person) or `LETTERBOX` (center crop)
4. **Crop Box Pre-computation** — Bounding boxes stored in the manifest for the Renderer

**Output:** Enriched manifest with `"crop"` key per clip — no video files written.

---

### SubsM — Subtitle Transcription

Enriches the manifest with word-level transcription data for animated subtitle rendering.

**How it works:**

1. **Audio Extraction** — FFmpeg extracts the audio segment for each clip
2. **Whisper Transcription** — PyCaps' WhisperAudioTranscriber provides word-level timestamps
3. **Timestamp Mapping** — Word times are offset back to source video coordinates

**Output:** Enriched manifest with `"subs"` key per clip (word list, template config) — no video files written.

---

### RendererM — Single-Pass Compositor

The only module that touches video files. Reads the enriched manifest and produces final clips.

**For each clip:**

1. **Cut** — FFmpeg extracts the segment from the source (stream copy, fast)
2. **Crop** — If `"crop"` key is present, applies CropperM's full rendering pipeline
3. **Subtitles** — If `"subs"` key is present, burns animated subtitles via PyCaps
4. **Output** — Writes the final `.mp4` file and cleans up temp files

**Module toggling:** The Renderer checks for key *presence* in the manifest, not boolean flags. If you disable CropperM, there's no `"crop"` key, so no cropping happens — zero code changes needed.

---

## 📋 Scenarios

Scenarios are pre-built configurations that define the entire pipeline behavior. Each scenario is a Python file with a `get_*_config()` function that returns a configuration dictionary.

### TikTok Scenario (`tiktok.py`)

Optimized for short, punchy, high-energy vertical clips.

| Parameter | Value | Rationale |
|---|---|---|
| Clip duration | 15–90s | Short attention span format |
| Max clips | 20 | Higher volume |
| Aspect ratio | 9:16 | Vertical mobile format |
| Subtitle position | 75% down | TikTok-optimized placement |
| Quality | High | Premium output |

```bash
python src/scenarios/tiktok.py --input "video.mp4" --clips 10
```

### Podcast Scenario (`podcast.py`)

Optimized for longer, conversational, story-driven clips.

| Parameter | Value | Rationale |
|---|---|---|
| Clip duration | 45–90s | Longer stories need room to breathe |
| Max clips | 10 | Quality over quantity |
| Aspect ratio | 9:16 | YouTube Shorts / Reels |
| Subtitle position | 70% down | Standard positioning |
| Quality | Balanced | Good quality, faster encoding |

```bash
python src/scenarios/podcast.py --input "podcast.mp4" --clips 10
```

### Gaming Scenario (`gaming_pipeline.py`)

The ultimate AI-powered auto-clipper for gameplay videos (e.g., Apex Legends, Valorant). Uses a synchronized multi-stage architecture: **Scanner → Audio Score → Clumper → Pacing Cushion → Renderer**.

| Stage | Module | Description |
|---|---|---|
| 🔊 **Audio** | AudioAnalyzerM | Scores combat intensity (spikes) for fast search pruning |
| 🎯 **Scanner** | DetectionM | Cross-scans YOLO (kills, knocks) & OCR (Victory feeds) on audio-active windows |
| 🎬 **Clumper** | Clipper_gamingM | Bundles frames & audio anchor intervals with smart merges into clips |
| 🔍 **Pacing** | SafeEntryExit | Fine-tunes cut points with exact safety padding buffers (VLM or Audio based) |
| 🎥 **Renderer** | RendererM | Single-pass compositing supporting fast **NVIDIA NVENC** Hardware Accel |

**To run the full end-to-end pipeline:**
```bash
python src/scenarios/gaming_pipeline.py --input "gameplay.mp4" --model "Train/runs/detect/train/weights/best.pt" --game "apex_legends"
```

**Options:**
- `--game`: Choose game pre-config: `apex_legends`, `valorant`, `fortnite`, `custom`.
- `--model`: Your custom trained model weights (e.g. `.pt` from Label Studio updates).
- `--window`: Clumping scanning bounds rate buffer (default 60s).
- `--min_intensity`: Lower scores filter boundary threshold (0-10 max bounds).
- `--no_safe_entry_exit`: Skip VLM/Audio cushioning for raw cuts.
- `--no_render`: Disable clip output writes for rapid dry-runs.

> [!TIP]
> Use the scripts in the `Train/` directory for manual scrubbing (`scrub_frames.py`) or automated frame extraction (`extract_frames.py`) if you are building a custom dataset.

---

## ⚙️ Full Configuration Reference

Every parameter is user-controllable in the scenario config. Here is the complete reference:

### Clipper Module

```python
"clipper": {
    "enabled": True,                          # Toggle module on/off

    # LLM Configuration
    "lm_studio_url": "http://localhost:1234/v1",
    "scout_model": "deepseek-r1-distill-qwen-32b",
    "editor_model": "google/gemma-3-27b",

    # Clip Parameters
    "min_clip_duration": 15.0,                # Seconds
    "max_clip_duration": 90.0,                # Seconds
    "max_total_clips": 20,

    # Analysis Features
    "enable_ocr": False,                      # EasyOCR for on-screen text
    "enable_vision": True,                    # Multimodal VLM keyframe analysis
    "vision_model": "qwen/qwen3-vl-30b",
    "vision_interval": 2.0,                   # Seconds between sampled frames
    "vision_concurrency": 6,                  # Parallel vision API calls

    # Quality Control
    "deduplication_threshold": 0.5,           # 0.0 = aggressive, 1.0 = permissive

    # Viral Archetypes (diversity enforcement)
    "viral_archetypes": [
        "High-Stakes Challenge",
        "Mind-Blowing Fact",
        "Hilarious/Raw Reaction",
        "Hot Take / Debate",
        "Satisfying Process"
    ],

    # LLM Prompts (fully customizable)
    "scout_system_instruction": "...",
    "scout_user_prompt": "...",
    "editor_system_instruction": "...",
    "editor_user_prompt": "..."
}
```

### Cropper Module

```python
"cropper": {
    "enabled": True,                          # Toggle module on/off
    "ratio": "9:16",                          # Target aspect ratio
    "quality": "balanced",                    # balanced, high, or fast
    "encoder": "auto",                        # auto, h264, hevc, or specific codec
    "crf": None,                              # Override CRF (lower = higher quality)
    "preset": None,                           # Override encoding preset
    "frame_skip": 0,                          # Skip N frames in scene detection
    "downscale": 0                            # Downscale factor for scene detection
}
```

### Subs Module

```python
"subs": {
    "enabled": True,                          # Toggle module on/off
    "template": "templates/hype",             # PyCaps subtitle template path
    "vertical_align_offset": 0.75,            # 0.0 = top, 1.0 = bottom
    "max_width_ratio": 0.85,                  # Max text width as ratio of video width
    "max_lines": 2,                           # Maximum simultaneous subtitle lines
    "whisper_model": "base",                  # tiny, base, small, medium, large
    "whisper_language": "en"                   # Language code for transcription
}
```

### Renderer Module

```python
"renderer": {
    "enabled": True,                          # Toggle module on/off
    "encoder": "auto",                        # Encoding codec
    "quality": "high",                        # Encoding quality preset
    "crf": 20,                                # Constant Rate Factor
    "preset": "slow"                          # Encoding speed/quality tradeoff
}
```

---

## 🎨 Subtitle Templates

ClipperAI ships with 14 built-in PyCaps subtitle templates:

| Template | Style |
|---|---|
| `default` | Clean, minimal white text |
| `hype` | Bold, high-energy with color accents |
| `explosive` | Large impact text with animations |
| `vibrant` | Colorful, dynamic styling |
| `minimalist` | Ultra-clean, thin weight |
| `neo-minimal` | Modern minimal with subtle effects |
| `classic` | Traditional subtitle appearance |
| `fast` | Optimized for rapid rendering |
| `line-focus` | Highlights the current line |
| `word-focus` | Highlights the current word |
| `submagic-impact` | SubMagic-inspired bold look |
| `retro-gaming` | Pixel/gaming aesthetic |
| `model` | Reference/base template |
| `testing` | Development use |

Set the template in your scenario config:
```python
"subs": {
    "template": "templates/hype"  # Use the hype template
}
```

---

## 🔧 Module Toggling

Any module can be independently enabled or disabled in the scenario config. The pipeline automatically adapts:

```python
# Full pipeline (default)
"clipper":  {"enabled": True},
"cropper":  {"enabled": True},
"subs":     {"enabled": True},
"renderer": {"enabled": True},

# Clips only — just find moments, no rendering
"cropper":  {"enabled": False},
"subs":     {"enabled": False},
"renderer": {"enabled": False},

# No subtitles
"subs":     {"enabled": False},

# No cropping — keep original aspect ratio
"cropper":  {"enabled": False},

# Plan only — generate manifest for external tools
"renderer": {"enabled": False},
```

When a module is disabled, its key won't appear in the manifest, and the Renderer automatically skips that transform. No code changes, no conditional logic — just configuration.

---

## 🧠 LLM Setup (LM Studio)

ClipperAI uses [LM Studio](https://lmstudio.ai) for local LLM inference. You need to have it running before starting the pipeline.

### Required Models

| Role | Recommended Model | Purpose |
|---|---|---|
| **Scout** | `deepseek-r1-distill-qwen-32b` | Reasoning model for clip identification |
| **Editor** | `google/gemma-3-27b` | Creative model for curation and titling |
| **Vision** (optional) | `qwen/qwen3-vl-30b` | Multimodal model for keyframe descriptions |

### LM Studio Configuration Tips

| Setting | Recommendation | Why |
|---|---|---|
| GPU Offload | Maximum layers | Faster inference |
| Context Length | 8192–16384 | Prompts are short; saves VRAM |
| Flash Attention | Enabled | Performance boost |
| KV Cache Quantization | Q4.0 | Saves VRAM |
| `vision_concurrency` | 4–6 | Parallel frame analysis |
| `vision_interval` | 2.0–5.0s | Fewer frames = faster with minimal quality loss |

---

## 📊 Output Structure

```
tiktok_output/
├── 1_push_up_challenge.mp4         # Final rendered clip
├── 2_rock_paper_scissors.mp4       # Final rendered clip
├── 3_blindfold_surprise.mp4        # Final rendered clip
├── clips_manifest.json             # Full manifest (resumable checkpoint)
└── clips_metadata.json             # Final metadata for rendered clips
```

### Manifest Schema

Each clip in `clips_manifest.json` contains:

```json
{
    "clip_id": "scout_1_clip_1",
    "title": "$1,000 Per Push-Up!",
    "file_name": "1_1000_per_pushup.mp4",
    "start": 45.2,
    "end": 112.8,
    "duration": 67.6,
    "scores": {
        "hook_strength": 95,
        "payoff_satisfaction": 90,
        "retention_potential": 88
    },
    "crop": { "...scene data..." },
    "subs": { "...word timestamps..." }
}
```

---

## 📈 Performance Tips

| Bottleneck | Solution |
|---|---|
| **LLM analysis slow** | Increase `vision_concurrency`, reduce `vision_interval`, lower LM Studio context length |
| **Scene detection slow** | Increase `frame_skip` or `downscale` in cropper config |
| **Whisper transcription slow** | Use `"whisper_model": "tiny"` or `"base"` instead of `"large"` |
| **Rendering slow** | Use `"quality": "fast"` or `"preset": "ultrafast"` in renderer config |
| **VRAM exhaustion** | Lower LM Studio context length, use smaller models, reduce `vision_concurrency` |
| **Disk I/O** | Use an SSD for the output directory |

---

## 🐛 Troubleshooting

### Common Issues

**Pipeline fails with `FileNotFoundError` for FFmpeg:**
Ensure FFmpeg binaries are in the `bin/` directory or on your system PATH.

**`Object of type int32 is not JSON serializable`:**
This is handled automatically by the pipeline's NumPy-aware JSON encoder. If you see this, ensure you're running the latest version.

**`[h264 @ ...] mmco: unref short failure`:**
Harmless FFmpeg warning from the H.264 decoder. Does not affect output quality.

**LM Studio connection refused:**
Ensure LM Studio is running on `http://localhost:1234` with a model loaded. Check the `lm_studio_url` in your scenario config.

**No clips found:**
The Scout model didn't find moments meeting the quality threshold. Try lowering `deduplication_threshold`, increasing clip duration range, or adjusting the `scout_system_instruction` prompt.

---

## 🤝 Contributing

Contributions are welcome! The modular architecture makes it straightforward to:

- **Add new scenarios** — Create a new file in `src/scenarios/` with a `get_*_config()` function
- **Add new modules** — Create a module in `src/modules/` with a `plan()` function (enriches manifest) and/or `main()` function
- **Add new subtitle templates** — Create a folder in `templates/` with PyCaps template files
- **Improve LLM prompts** — Modify `scout_system_instruction` / `editor_system_instruction` in scenario configs

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [LM Studio](https://lmstudio.ai) — Local LLM inference
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech-to-text transcription
- [PyCaps](https://github.com/francozanardi/pycaps) — Animated subtitle rendering
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Person detection
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — Scene analysis
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — On-screen text extraction
- [FFmpeg](https://ffmpeg.org) — Media encoding and decoding
- [MoviePy](https://github.com/Zulko/moviepy) — Video processing utilities

---

<div align="center">

**Built for content creators who want to ship more, faster.**

[⭐ Star this repo](https://github.com/Preethum/ClipperAI) · [🍴 Fork](https://github.com/Preethum/ClipperAI/fork) · [🐛 Issues](https://github.com/Preethum/ClipperAI/issues)

</div>
