# 🎬 ClipperAI

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/yourusername/ClipperAI)
[![Contributors](https://img.shields.io/badge/contributors-welcome-orange.svg)](CONTRIBUTING.md)

> 🤖 **AI-Powered Video Clipping System** - Automatically extract viral clips from long-form videos using advanced AI analysis and multi-agent processing.

## ✨ Features

- 🧠 **Multi-Agent AI Analysis** - Scout + Editor pipeline for intelligent clip selection
- 🎯 **Smart Content Detection** - Identifies viral moments, hooks, and payoff points
- 📱 **Vertical Format Optimization** - Automatic 9:16 cropping for social media
- 📝 **Dynamic Subtitle Generation** - AI-powered transcription with styled overlays
- 🔧 **Modular Architecture** - Extensible pipeline with customizable scenarios
- ⚡ **High-Performance Processing** - GPU-accelerated transcription and analysis
- 🎨 **Template System** - Customizable subtitle templates and styling
- 📊 **Quality Scoring** - Hybrid virality metrics for optimal clip selection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended for faster processing)
- FFmpeg (included in `/bin` directory)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ClipperAI.git
cd ClipperAI

# Create virtual environment
python -m venv clipper-venv
source clipper-venv/bin/activate  # On Windows: clipper-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process a video with default settings
python src/scenarios/podcast.py --input "your_video.mp4" --output "clips" --clips 10

# Process MrBeast-style content
python src/scenarios/podcast.py --input "mrbeast_video.mp4" --output "viral_clips" --clips 15

# Process podcast content
python src/scenarios/podcast.py --input "podcast_episode.mp4" --output "podcast_clips" --clips 20
```

## 📁 Project Structure

```
ClipperAI/
├── 📂 src/
│   ├── 📂 modules/              # Core processing modules
│   │   ├── 🐍 ClipperM.py       # AI-powered clip extraction
│   │   ├── ✂️ CropperM.py       # Smart video cropping
│   │   └── 📝 SubsM.py          # Subtitle generation
│   ├── 📂 scenarios/           # Video processing scenarios
│   │   └── 🎙️ podcast.py        # Podcast processing pipeline
│   └── 📂 utils/
│       └── 🛠️ path_utils.py     # Path resolution utilities
├── 📂 bin/                      # External binaries
│   ├── 🎬 ffmpeg.exe
│   ├── 🎮 ffplay.exe
│   └── 🔍 ffprobe.exe
├── 📂 templates/               # Subtitle templates
├── 📂 output/                  # Generated clips
├── 📋 requirements.txt
├── 📄 LICENSE
├── 📖 README.md
└── 🚫 .gitignore
```

## 🎯 Processing Pipeline

### 1. **Clipper Module** - AI Analysis
- **Transcription**: Whisper-based speech-to-text
- **OCR Analysis**: Visual content extraction
- **Semantic Chapters**: Content segmentation
- **Multi-Agent Selection**: Scout + Editor pipeline
- **Virality Scoring**: Hybrid metrics (narrative + visual + audio)

### 2. **Cropper Module** - Format Optimization
- **Smart Cropping**: Face detection and tracking
- **Aspect Ratio**: 9:16 vertical format
- **Quality Control**: Multiple encoding presets
- **Scene Analysis**: Optimal framing selection

### 3. **Subs Module** - Subtitle Generation
- **Dynamic Styling**: Template-based rendering
- **Positioning**: Smart vertical alignment
- **Font Scaling**: Responsive text sizing
- **Template System**: Customizable designs

## ⚙️ Configuration

### Default Settings

```python
{
    "modules": {
        "clipper": {
            "min_clip_duration": 45.0,      # Minimum clip length (seconds)
            "max_clip_duration": 90.0,      # Maximum clip length (seconds)
            "max_total_clips": 10,           # Maximum number of clips
            "scout_model": "deepseek-r1-distill-qwen-32b",
            "editor_model": "google/gemma-3-27b",
            "lm_studio_url": "http://localhost:1234/v1"
        },
        "cropper": {
            "ratio": "9:16",                 # Target aspect ratio
            "quality": "balanced",           # Encoding quality
            "encoder": "auto"                # Auto-select best encoder
        },
        "subs": {
            "template": "hype",              # Subtitle template
            "vertical_align_offset": 0.70,   # Vertical positioning
            "max_width_ratio": 0.9,          # Maximum text width
            "max_lines": 1                   # Maximum subtitle lines
        }
    }
}
```

### Command Line Options

```bash
python src/scenarios/podcast.py [OPTIONS]

Required:
  --input TEXT        Input video file path
  --output TEXT       Output directory path

Optional:
  --clips INTEGER     Maximum number of clips (default: 10)
  --config TEXT       Path to JSON configuration file
  --min-duration FLOAT Minimum clip duration (default: 45.0)
  --max-duration FLOAT Maximum clip duration (default: 90.0)
```

## 📊 Output Structure

```
output/
├── 📂 01_clips/              # Clipper output
│   ├── 🎬 viral_clip_1.mp4
│   ├── 🎬 viral_clip_2.mp4
│   └── 📊 clips_metadata.json
├── 📂 02_cropped/            # Cropper output
│   ├── 📱 viral_clip_1.mp4
│   └── 📱 viral_clip_2.mp4
└── 📂 03_final/              # Final output with subtitles
    ├── 🎥 viral_clip_1.mp4
    └── 🎥 viral_clip_2.mp4
```

## 🎨 Subtitle Templates

### Available Templates
- **hype** - High-energy, bold styling
- **default** - Clean, minimal design
- **custom** - Your custom templates

### Creating Custom Templates

1. Navigate to `/templates/` directory
2. Create your template folder
3. Add subtitle configuration files
4. Reference in config: `"template": "your_template_name"`

## 🧠 AI Models

### Supported Models
- **Scout Model**: `deepseek-r1-distill-qwen-32b` - Content analysis and filtering
- **Editor Model**: `google/gemma-3-27b` - Creative selection and refinement
- **Transcription**: Whisper (small, medium, large models)
- **OCR**: EasyOCR with GPU acceleration

### Model Configuration
```python
# LM Studio integration
"lm_studio_url": "http://localhost:1234/v1"

# Whisper settings
whisper_model = "small"  # Options: tiny, base, small, medium, large
device = "cuda"           # GPU acceleration
compute_type = "float16"  # Precision optimization
```

## 🔧 Advanced Usage

### Custom Configuration File

Create a JSON config file:

```json
{
    "input_video": "custom_video.mp4",
    "base_output_dir": "custom_output",
    "modules": {
        "clipper": {
            "max_total_clips": 25,
            "min_clip_duration": 30.0,
            "max_clip_duration": 120.0
        },
        "cropper": {
            "ratio": "16:9",
            "quality": "high"
        },
        "subs": {
            "template": "custom_template",
            "max_lines": 2
        }
    }
}
```

Run with custom config:
```bash
python src/scenarios/podcast.py --config custom_config.json
```

### Batch Processing

```bash
# Process multiple videos
for video in *.mp4; do
    python src/scenarios/podcast.py --input "$video" --output "clips_$(basename "$video" .mp4)" --clips 15
done
```

## 🛠️ Development

### Adding New Modules

1. Create module in `src/modules/`
2. Implement required interface
3. Register in scenario pipeline
4. Add configuration options

### Adding New Scenarios

1. Create scenario file in `src/scenarios/`
2. Define processing pipeline
3. Configure module settings
4. Add command-line interface

### Testing

```bash
# Run tests
python -m pytest tests/

# Test with sample video
python src/scenarios/podcast.py --input "sample_video.mp4" --output "test_output" --clips 3
```

## 🐛 Troubleshooting

### Common Issues

**❌ Bin directory not found**
```bash
# Ensure FFmpeg binaries are in /bin directory
ls bin/
# Should show: ffmpeg.exe, ffplay.exe, ffprobe.exe
```

**❌ CUDA out of memory**
```python
# Reduce model size in ClipperM.py
whisper_model = "base"  # Use smaller model
```

**❌ LM Studio connection failed**
```bash
# Ensure LM Studio is running on port 1234
# Check model availability in LM Studio
```

**❌ Template not found**
```bash
# Verify template exists in templates/ directory
ls templates/
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### GPU Requirements
- **VRAM**: 8GB+ recommended for large models
- **CUDA**: 11.0+ for optimal performance
- **Memory**: 16GB+ system RAM

### Speed Tips
- Use GPU acceleration (`device="cuda"`)
- Choose appropriate model sizes
- Optimize clip duration ranges
- Use SSD for video I/O

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ClipperAI.git
cd ClipperAI

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** - Whisper transcription model
- **EasyOCR** - Text extraction from video frames
- **LM Studio** - Local LLM inference
- **MoviePy** - Video processing
- **FFmpeg** - Media encoding/decoding

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/ClipperAI/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/ClipperAI/discussions)
- 📧 **Email**: your-email@example.com

---

<div align="center">

**🎬 Built with ❤️ for content creators and social media managers**

[⭐ Star this repo](https://github.com/yourusername/ClipperAI) • [🍴 Fork](https://github.com/yourusername/ClipperAI/fork) • [📖 Documentation](https://github.com/yourusername/ClipperAI/wiki)

</div>
