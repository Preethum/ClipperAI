"""
Gaming scenario configuration and interface.
This file uses the run_gaming_pipeline global function and holds all gaming-specific configurations.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from global_functions import run_gaming_pipeline
from typing import Dict, List, Any


class GameConfig:
    """Base class for game-specific configurations"""
    
    def __init__(self, game_name: str, config_data: Dict = None):
        self.game_name = game_name
        self.config_data = config_data or self.get_default_config()
        
    def get_default_config(self) -> Dict:
        """Get default configuration template"""
        return {
            "game_info": {
                "name": self.game_name,
                "description": "Generic game configuration",
                "platform": "pc",
                "genre": "action"
            },
            
            # Detection Configuration
            "detection": {
                "model_path": "yolov8m.pt",
                "interval_seconds": 1.0,
                "min_confidence": 0.5,
                "enable_ocr": True,
                "debug": False,
                "target_labels": [],
                "label_weights": {}
            },
            
            # Game-specific scoring system
            "scoring": {
                "points_system": {
                    "elimination": 100,
                    "victory": 500,
                    "damage": 10,
                    "special_event": 200
                },
                "multipliers": {
                    "streak_bonus": 1.5,
                    "accuracy_bonus": 1.2,
                    "speed_bonus": 1.1
                }
            },
            
            # Audio Configuration
            "audio": {
                "enabled": True,
                "device": 0,
                "chunk_sec": 1.0,
                "threshold": 0.30,
                "merge_to_events": True
            },
            
            # Clumping Configuration
            "clumping": {
                "mode": "elimination",
                "elim_merge_gap": 30.0,
                "audio_lookback": 30.0,
                "audio_lookahead": 30.0,
                "end_buffer": 5.0,
                "clip_merge_gap": 60.0,
                "window_seconds": 60,
                "min_intensity": 3.0,
                "gap_threshold": 30.0,
                "intensity_weights": {}
            },
            
            # Safe Entry/Exit Configuration
            "safe_entry_exit": {
                "enabled": True,
                "vlm_model": "google/gemma-3-27b",
                "vlm_fps": 10,
                "entry_buffer": 1.2,
                "exit_buffer": 1.5
            },
            
            # Rendering Configuration
            "rendering": {
                "enabled": True,
                "format": "mp4",
                "encoder": "auto",
                "quality": "balanced",
                "crf": None,
                "preset": None
            }
        }
    
    def get_intensity_weights(self) -> Dict:
        """Get intensity weights for event scoring"""
        return self.config_data.get("clumping", {}).get("intensity_weights", {
            'base_score': 1.0,
            'combat_bonus': 2.0,
            'high_confidence_bonus': 0.5,
            'squad_wipe_bonus': 1.5,
            'high_damage_bonus': 1.0,
            'elimination_bonus': 1.0,
            'label_bonuses': {}
        })
    
    def get_target_labels(self) -> List[str]:
        """Get target labels for detection"""
        return self.config_data.get("detection", {}).get("target_labels", [])
    
    def get_scoring_points(self) -> Dict:
        """Get points system for game events"""
        return self.config_data.get("scoring", {}).get("points_system", {})
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return self.config_data


class ApexLegendsConfig(GameConfig):
    """Apex Legends specific configuration"""
    
    def __init__(self):
        super().__init__("apex_legends")
        self.config_data = self._get_apex_config()
    
    def _get_apex_config(self) -> Dict:
        base_config = self.get_default_config()
        
        # Update with Apex Legends specific settings
        base_config.update({
            "game_info": {
                "name": "apex_legends",
                "description": "Apex Legends battle royale game",
                "platform": "pc",
                "genre": "battle_royale"
            },
            
            "detection": {
                "model_path": "yolov8m.pt",
                "interval_seconds": 1.0,
                "min_confidence": 0.5,
                "enable_ocr": True,
                "debug": False,
                "target_labels": [
                    "Victory", 
                    "ELIMINATED Text", 
                    "Floating Damage Numbers", 
                    "Directional Damage Indicators",
                    "Downed Player",
                    "Revive Indicator",
                    "Supply Drop"
                ],
                "label_weights": {
                    "Victory": 5.0,
                    "ELIMINATED Text": 4.0,
                    "Floating Damage Numbers": 2.0,
                    "Directional Damage Indicators": 1.5,
                    "Downed Player": 3.0,
                    "Revive Indicator": 2.5,
                    "Supply Drop": 1.0
                }
            },
            
            "scoring": {
                "points_system": {
                    "elimination": 100,
                    "victory": 1000,
                    "damage": 5,
                    "down": 50,
                    "revive": 75,
                    "supply_drop": 25,
                    "squad_wipe": 500
                },
                "multipliers": {
                    "streak_bonus": 1.5,
                    "accuracy_bonus": 1.2,
                    "speed_bonus": 1.1,
                    "squad_bonus": 2.0
                }
            },
            
            "clumping": {
                "mode": "elimination",
                "elim_merge_gap": 30.0,
                "audio_lookback": 30.0,
                "audio_lookahead": 30.0,
                "end_buffer": 5.0,
                "clip_merge_gap": 60.0,
                "window_seconds": 60,
                "min_intensity": 3.0,
                "gap_threshold": 30.0,
                "intensity_weights": {
                    'base_score': 1.0,
                    'combat_bonus': 2.0,
                    'high_confidence_bonus': 0.5,
                    'squad_wipe_bonus': 1.5,
                    'high_damage_bonus': 1.0,
                    'elimination_bonus': 1.0,
                    'label_bonuses': {
                        'Victory': 10.0,
                        'ELIMINATED Text': 8.0,
                        'Floating Damage Numbers': 3.0,
                        'Directional Damage Indicators': 2.0,
                        'Downed Player': 6.0,
                        'Revive Indicator': 4.0,
                        'Supply Drop': 1.5
                    }
                }
            }
        })
        
        return base_config


class ValorantConfig(GameConfig):
    """Valorant specific configuration"""
    
    def __init__(self):
        super().__init__("valorant")
        self.config_data = self._get_valorant_config()
    
    def _get_valorant_config(self) -> Dict:
        base_config = self.get_default_config()
        
        base_config.update({
            "game_info": {
                "name": "valorant",
                "description": "Valorant tactical shooter game",
                "platform": "pc",
                "genre": "tactical_shooter"
            },
            
            "detection": {
                "model_path": "yolov8m.pt",
                "interval_seconds": 0.5,  # Faster for tactical shooter
                "min_confidence": 0.6,
                "enable_ocr": True,
                "debug": False,
                "target_labels": [
                    "Victory", 
                    "Defeat", 
                    "Eliminated Text",
                    "Headshot Indicator",
                    "Ability Usage",
                    "Spike Plant",
                    "Spike Defuse",
                    "Round Win"
                ],
                "label_weights": {
                    "Victory": 5.0,
                    "Eliminated Text": 4.0,
                    "Headshot Indicator": 3.5,
                    "Ability Usage": 2.0,
                    "Spike Plant": 4.0,
                    "Spike Defuse": 4.5,
                    "Round Win": 3.0
                }
            },
            
            "scoring": {
                "points_system": {
                    "elimination": 100,
                    "headshot": 150,
                    "victory": 300,
                    "defeat": -50,
                    "spike_plant": 75,
                    "spike_defuse": 100,
                    "ability_kill": 125,
                    "ace": 500,
                    "clutch": 400
                },
                "multipliers": {
                    "streak_bonus": 1.3,
                    "accuracy_bonus": 1.4,
                    "speed_bonus": 1.0,
                    "tactical_bonus": 1.5
                }
            },
            
            "clumping": {
                "mode": "sliding_window",  # Better for round-based gameplay
                "window_seconds": 120,  # Longer rounds
                "min_intensity": 2.5,
                "gap_threshold": 20.0,
                "intensity_weights": {
                    'base_score': 1.0,
                    'combat_bonus': 2.5,
                    'high_confidence_bonus': 0.5,
                    'squad_wipe_bonus': 2.0,
                    'high_damage_bonus': 1.2,
                    'elimination_bonus': 1.5,
                    'label_bonuses': {
                        'Victory': 8.0,
                        'Eliminated Text': 6.0,
                        'Headshot Indicator': 5.0,
                        'Ability Usage': 3.0,
                        'Spike Plant': 5.0,
                        'Spike Defuse': 6.0,
                        'Round Win': 4.0
                    }
                }
            }
        })
        
        return base_config


class FortniteConfig(GameConfig):
    """Fortnite specific configuration"""
    
    def __init__(self):
        super().__init__("fortnite")
        self.config_data = self._get_fortnite_config()
    
    def _get_fortnite_config(self) -> Dict:
        base_config = self.get_default_config()
        
        base_config.update({
            "game_info": {
                "name": "fortnite",
                "description": "Fortnite battle royale game",
                "platform": "pc",
                "genre": "battle_royale"
            },
            
            "detection": {
                "model_path": "yolov8m.pt",
                "interval_seconds": 1.0,
                "min_confidence": 0.5,
                "enable_ocr": True,
                "debug": False,
                "target_labels": [
                    "Victory Royale", 
                    "Eliminated Text", 
                    "Damage Numbers",
                    "Building Indicator",
                    "Storm Circle",
                    "Supply Drop",
                    "Legendary Item"
                ],
                "label_weights": {
                    "Victory Royale": 5.0,
                    "Eliminated Text": 4.0,
                    "Damage Numbers": 2.0,
                    "Building Indicator": 2.5,
                    "Storm Circle": 1.5,
                    "Supply Drop": 2.0,
                    "Legendary Item": 3.0
                }
            },
            
            "scoring": {
                "points_system": {
                    "elimination": 100,
                    "victory_royale": 1000,
                    "damage": 3,
                    "build_destroy": 50,
                    "supply_drop": 30,
                    "legendary_item": 75,
                    "storm_survival": 25
                },
                "multipliers": {
                    "streak_bonus": 1.6,
                    "build_bonus": 1.3,
                    "accuracy_bonus": 1.2,
                    "survival_bonus": 1.4
                }
            }
        })
        
        return base_config


# Game configuration registry
GAME_CONFIGS = {
    "apex_legends": ApexLegendsConfig,
    "valorant": ValorantConfig,
    "fortnite": FortniteConfig,
    "custom": GameConfig
}


def get_game_config(game_name: str, config_path: str = None) -> GameConfig:
    """Get game configuration by name or from file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        game_name = config_data.get("game_info", {}).get("name", "custom")
        return GameConfig(game_name, config_data)
    
    if game_name in GAME_CONFIGS:
        return GAME_CONFIGS[game_name]()
    else:
        # Return custom config with provided name
        return GameConfig(game_name)


def list_available_games() -> List[str]:
    """List all available game configurations"""
    return list(GAME_CONFIGS.keys())

def get_gaming_config():
    """Get default configuration for gaming highlight processing."""
    # Use Apex Legends as default game configuration
    game_config = get_game_config("apex_legends")
    return convert_game_config_to_pipeline_config(game_config)

def convert_game_config_to_pipeline_config(game_config):
    """Convert game configuration to pipeline configuration format"""
    game_data = game_config.to_dict()
    
    return {
        # Input/Output paths
        "input_video": "source.mkv",
        "base_output_dir": "output",
        
        # Game information
        "game_info": game_data.get("game_info", {}),
        
        # Scoring system
        "scoring": game_data.get("scoring", {}),
        
        # Module configuration
        "modules": {
            "detector": {
                "enabled": True,
                
                # Detection settings from game config
                "model_path": game_data.get("detection", {}).get("model_path", "yolov8m.pt"),
                "interval_seconds": game_data.get("detection", {}).get("interval_seconds", 1.0),
                "min_confidence": game_data.get("detection", {}).get("min_confidence", 0.5),
                "enable_ocr": game_data.get("detection", {}).get("enable_ocr", True),
                "debug": game_data.get("detection", {}).get("debug", False),
                
                # Target labels from game config
                "target_labels": game_data.get("detection", {}).get("target_labels", []),
                "label_weights": game_data.get("detection", {}).get("label_weights", {})
            },
            
            "audio": {
                "enabled": game_data.get("audio", {}).get("enabled", True),
                "device": game_data.get("audio", {}).get("device", 0),
                "chunk_sec": game_data.get("audio", {}).get("chunk_sec", 1.0),
                "threshold": game_data.get("audio", {}).get("threshold", 0.30),
                "merge_to_events": game_data.get("audio", {}).get("merge_to_events", True)
            },
            
            "clumper": {
                "enabled": True,
                "mode": game_data.get("clumping", {}).get("mode", "elimination"),
                "elim_merge_gap": game_data.get("clumping", {}).get("elim_merge_gap", 30.0),
                "audio_lookback": game_data.get("clumping", {}).get("audio_lookback", 30.0),
                "audio_lookahead": game_data.get("clumping", {}).get("audio_lookahead", 30.0),
                "end_buffer": game_data.get("clumping", {}).get("end_buffer", 5.0),
                "clip_merge_gap": game_data.get("clumping", {}).get("clip_merge_gap", 60.0),
                "window_seconds": game_data.get("clumping", {}).get("window_seconds", 60),
                "min_intensity": game_data.get("clumping", {}).get("min_intensity", 3.0),
                "gap_threshold": game_data.get("clumping", {}).get("gap_threshold", 30.0),
                "intensity_weights": game_config.get_intensity_weights()
            },
            
            "safe_entry_exit": {
                "enabled": game_data.get("safe_entry_exit", {}).get("enabled", True),
                "vlm_model": game_data.get("safe_entry_exit", {}).get("vlm_model", "google/gemma-3-27b"),
                "vlm_fps": game_data.get("safe_entry_exit", {}).get("vlm_fps", 10),
                "entry_buffer": game_data.get("safe_entry_exit", {}).get("entry_buffer", 1.2),
                "exit_buffer": game_data.get("safe_entry_exit", {}).get("exit_buffer", 1.5)
            },
            
            "renderer": {
                "enabled": game_data.get("rendering", {}).get("enabled", True),
                "format": game_data.get("rendering", {}).get("format", "mp4"),
                "encoder": game_data.get("rendering", {}).get("encoder", "auto"),
                "quality": game_data.get("rendering", {}).get("quality", "balanced"),
                "crf": game_data.get("rendering", {}).get("crf", None),
                "preset": game_data.get("rendering", {}).get("preset", None)
            }
        },
        
        # Global Scenario Settings
        "scenario_name": "gaming",
        "target_platform": "tiktok_shorts",
        "content_style": "high_intensity_highlights"
    }

def main():
    """Run the gaming scenario."""
    parser = argparse.ArgumentParser(description="Gaming Event Detection & Clumping Pipeline")
    
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--game", type=str, help="Game type (apex_legends, valorant, fortnite, custom)")
    parser.add_argument("--list_games", action="store_true", help="List all available game configurations")
    
    # Video and model overrides
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--model", required=True, help="YOLO model file (.pt)")
    parser.add_argument("--output", help="Output directory (auto-generated if not specified)")
    
    # Detection overrides
    parser.add_argument("--interval", type=float, help="Scan interval (seconds)")
    parser.add_argument("--min_confidence", type=float, help="Minimum confidence threshold")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--label_weights", type=str, help="Label keyword weights as JSON string")
    
    # Clumping overrides
    parser.add_argument("--window", type=int, help="Clumping window size in seconds")
    parser.add_argument("--min_intensity", type=float, help="Minimum intensity score for clips")
    parser.add_argument("--gap_threshold", type=float, help="Minimum gap in seconds to start new clip")
    parser.add_argument("--intensity_weights", type=str, help="Intensity weights as JSON string")
    
    # Safe entry/exit overrides
    parser.add_argument("--no_safe_entry_exit", action="store_true", help="Skip safe entry/exit detection")
    parser.add_argument("--vlm_model", help="VLM model for safe entry/exit")
    parser.add_argument("--vlm_fps", type=int, help="Frames per second for VLM analysis")
    parser.add_argument("--entry_buffer", type=float, help="Minutes before clip for entry analysis")
    parser.add_argument("--exit_buffer", type=float, help="Minutes after clip for exit analysis")
    
    # Audio analysis overrides
    parser.add_argument("--no_audio", action="store_true", help="Disable audio analysis")
    parser.add_argument("--audio_device", type=int, help="CUDA device for audio AST model (0=GPU0, -1=CPU)")
    parser.add_argument("--audio_threshold", type=float, help="Minimum audio combat score")
    
    # Video rendering overrides
    parser.add_argument("--no_render", action="store_true", help="Skip video rendering")
    
    args = parser.parse_args()
    
    # List available games if requested
    if args.list_games:
        print("Available game configurations:")
        for game in list_available_games():
            print(f"  - {game}")
        return 0
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f" Loaded configuration from: {args.config}")
    else:
        # Use game-specific configuration
        game_name = args.game or "apex_legends"
        game_config = get_game_config(game_name, args.config)
        config = convert_game_config_to_pipeline_config(game_config)
        print(f" Using {game_name} game configuration")
        print(f" Game: {config['game_info'].get('description', 'Unknown')}")
        print(f" Target labels: {len(config['modules']['detector']['target_labels'])} labels configured")
        print(f" Scoring system: {len(config['scoring'].get('points_system', {}))} events configured")
        
    # --- Override configuration with command line arguments ---
    
    # Paths
    if args.input: config["input_video"] = args.input
    if args.output: 
        config["base_output_dir"] = args.output
    else:
        config["base_output_dir"] = f"gaming_pipeline_{datetime.now().strftime('%m%d_%H%M')}"
        
    m = config.get("modules", {})
    
    # Detection
    if args.model: m["detector"]["model_path"] = args.model
    if args.interval is not None: m["detector"]["interval_seconds"] = args.interval
    if args.min_confidence is not None: m["detector"]["min_confidence"] = args.min_confidence
    if args.no_ocr: m["detector"]["enable_ocr"] = False
    if args.debug: m["detector"]["debug"] = True
    if args.label_weights:
        try:
            m["detector"]["label_weights"] = json.loads(args.label_weights)
        except Exception:
            print("Failed to parse label_weights JSON.")
            
    # Audio
    if args.no_audio: m["audio"]["enabled"] = False
    if args.audio_device is not None: m["audio"]["device"] = args.audio_device
    if args.audio_threshold is not None: m["audio"]["threshold"] = args.audio_threshold
    
    # Clumping
    if args.window is not None: m["clumper"]["window_seconds"] = args.window
    if args.min_intensity is not None: m["clumper"]["min_intensity"] = args.min_intensity
    if args.gap_threshold is not None: m["clumper"]["gap_threshold"] = args.gap_threshold
    if args.intensity_weights:
        try:
            m["clumper"]["intensity_weights"] = json.loads(args.intensity_weights)
        except Exception:
            print("Failed to parse intensity_weights JSON.")
            
    # Pacing
    if args.no_safe_entry_exit: m["safe_entry_exit"]["enabled"] = False
    if args.vlm_model: m["safe_entry_exit"]["vlm_model"] = args.vlm_model
    if args.vlm_fps is not None: m["safe_entry_exit"]["vlm_fps"] = args.vlm_fps
    if args.entry_buffer is not None: m["safe_entry_exit"]["entry_buffer"] = args.entry_buffer
    if args.exit_buffer is not None: m["safe_entry_exit"]["exit_buffer"] = args.exit_buffer
    
    # Renderer
    if args.no_render: m["renderer"]["enabled"] = False
    
    config["modules"] = m
    
    # Run the universal pipeline framework
    final_clips = run_gaming_pipeline(config)
    
    print(f"\n✅ Pipeline execution finished.")
    return 0 if final_clips else 1

if __name__ == "__main__":
    sys.exit(main())
