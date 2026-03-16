# Gaming Pipeline Configuration System

This document explains how to use the modular gaming pipeline configuration system.

## Overview

The gaming pipeline now supports modular game configurations, allowing users to:
- Select from predefined game configurations (Apex Legends, Valorant, Fortnite)
- Create custom game configurations
- Configure game-specific scoring systems
- Adjust detection parameters for different games

## Usage

### 1. List Available Games
```bash
python gaming_pipeline.py --list_games
```

### 2. Use a Predefined Game Configuration
```bash
# Apex Legends (default)
python gaming_pipeline.py --game apex_legends --input video.mp4 --model yolov8m.pt

# Valorant
python gaming_pipeline.py --game valorant --input video.mp4 --model yolov8m.pt

# Fortnite
python gaming_pipeline.py --game fortnite --input video.mp4 --model yolov8m.pt
```

### 3. Use Custom Configuration File
```bash
python gaming_pipeline.py --config configs/my_game_config.json --input video.mp4 --model yolov8m.pt
```

## Configuration Structure

Each game configuration contains the following sections:

### Game Info
```json
{
  "game_info": {
    "name": "game_name",
    "description": "Game description",
    "platform": "pc",
    "genre": "genre"
  }
}
```

### Detection Configuration
```json
{
  "detection": {
    "model_path": "yolov8m.pt",
    "interval_seconds": 1.0,
    "min_confidence": 0.5,
    "enable_ocr": true,
    "target_labels": ["Victory", "Eliminated Text"],
    "label_weights": {"Victory": 5.0, "Eliminated Text": 4.0}
  }
}
```

### Scoring System
```json
{
  "scoring": {
    "points_system": {
      "elimination": 100,
      "victory": 500,
      "damage": 10
    },
    "multipliers": {
      "streak_bonus": 1.5,
      "accuracy_bonus": 1.2
    }
  }
}
```

### Audio Configuration
```json
{
  "audio": {
    "enabled": true,
    "device": 0,
    "chunk_sec": 1.0,
    "threshold": 0.30,
    "merge_to_events": true
  }
}
```

### Clumping Configuration
```json
{
  "clumping": {
    "mode": "elimination",
    "window_seconds": 60,
    "min_intensity": 3.0,
    "intensity_weights": {
      "base_score": 1.0,
      "combat_bonus": 2.0,
      "label_bonuses": {"Victory": 5.0}
    }
  }
}
```

## Creating Custom Game Configurations

### Method 1: Copy and Modify Template
1. Copy `configs/custom_game_template.json`
2. Modify the values for your specific game
3. Use with `--config path/to/your/config.json`

### Method 2: Create Programmatically
```python
from game_configs import create_custom_game_config

# Create custom config based on Apex Legends
custom_config = create_custom_game_config("my_game", "apex_legends")
custom_config.config_data["game_info"]["description"] = "My Custom Game"
custom_config.save_to_file("my_game_config.json")
```

## Game-Specific Parameters

### Apex Legends
- **Mode**: elimination-based clumping
- **Key Events**: Victory, Eliminations, Damage Numbers, Supply Drops
- **Scoring**: High value for squad wipes and victories

### Valorant
- **Mode**: sliding window (better for round-based gameplay)
- **Key Events**: Headshots, Spike plants/defuses, Round wins
- **Scoring**: Bonus for tactical plays and clutches

### Fortnite
- **Mode**: elimination-based with building focus
- **Key Events**: Victory Royale, Building, Storm survival
- **Scoring**: Points for building and survival elements

## Parameter Tuning

### Detection Parameters
- `interval_seconds`: Lower for fast-paced games (0.5 for Valorant), higher for slower games
- `min_confidence`: Adjust based on model performance and game UI clarity
- `target_labels`: Add game-specific UI elements you want to detect

### Scoring Parameters
- Adjust `points_system` values based on what constitutes highlights in your game
- Use `multipliers` to reward specific playstyles (accuracy, speed, etc.)

### Clumping Parameters
- `mode`: "elimination" for elimination-based games, "sliding_window" for round-based games
- `window_seconds`: Longer for games with extended combat sequences
- `min_intensity`: Adjust based on how intense highlights should be

## Example Workflows

### Stream Setup for Different Games
```bash
# Apex Legends stream
python gaming_pipeline.py --game apex_legends --input apex_stream.mp4 --model apex_model.pt

# Valorant tournament
python gaming_pipeline.py --game valorant --input valorant_match.mp4 --model valorant_model.pt --min_intensity 2.0

# Custom game with specific settings
python gaming_pipeline.py --config configs/my_custom_game.json --input gameplay.mp4 --model custom_model.pt
```

### Testing New Configurations
1. Start with the template configuration
2. Test with a short video clip
3. Adjust `min_confidence` if detection is missing events
4. Tune `intensity_weights` if highlights are too short/long
5. Modify `scoring` to match what you consider highlights

## Troubleshooting

### No Events Detected
- Check `target_labels` match what appears in your game
- Lower `min_confidence` threshold
- Ensure `model_path` points to a trained model

### Highlights Too Long/Short
- Adjust `window_seconds` in clumping configuration
- Modify `audio_lookback`/`audio_lookahead` for better timing
- Tune `min_intensity` threshold

### Wrong Game Events Prioritized
- Update `label_weights` to emphasize important events
- Modify `scoring` points system
- Adjust `intensity_weights` for better event ranking
