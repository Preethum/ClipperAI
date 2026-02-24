import os
import sys
import json
import shutil
from pathlib import Path

# Set working directory to project root
def set_project_root_as_cwd():
    """Change working directory to project root."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels from src/scenarios to project root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Change to project root directory
    os.chdir(project_root)
    print(f"📁 Changed working directory to: {project_root}")
    return project_root

# Set project root as current working directory
PROJECT_ROOT = set_project_root_as_cwd()

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_bin_dir, setup_bin_path, get_templates_dir, get_output_dir
except ImportError:
    # Fallback if utils module not available
    def get_project_root():
        # More robust fallback - try multiple methods
        current_file = os.path.abspath(__file__)
        # We're in src/scenarios, so go up 2 levels to project root
        return os.path.dirname(os.path.dirname(current_file))
    def get_bin_dir():
        return os.path.join(get_project_root(), 'bin')
    def get_templates_dir():
        return os.path.join(get_project_root(), 'templates')
    def get_output_dir():
        return os.path.join(get_project_root(), 'output')
    def setup_bin_path():
        bin_dir = get_bin_dir()
        if os.path.exists(bin_dir):
            current_path = os.environ.get("PATH", "")
            if bin_dir not in current_path:
                os.environ["PATH"] = bin_dir + os.pathsep + current_path
            return bin_dir
        else:
            print(f"Warning: Bin directory not found at {bin_dir}")
            return None

# Add folder to PATH for external libraries (like FFmpeg)
LOCAL_BIN_DIR = setup_bin_path()
if LOCAL_BIN_DIR is None:
    LOCAL_BIN_DIR = get_bin_dir()  # For reference in error messages

# Add the modules directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from ClipperM import main as clipper_main
from CropperM import main as cropper_main
from SubsM import main as subs_main

class PodcastScenario:
    """
    Podcast video processing scenario that orchestrates ClipperM, CropperM, and SubsM
    in sequence to create viral podcast clips with subtitles.
    """
    
    def __init__(self, config=None):
        """
        Initialize the podcast scenario with configuration.
        
        Args:
            config (dict): Configuration dictionary containing all settings
        """
        self.config = config or self.get_default_config()
        self.temp_dirs = []
        
    def get_default_config(self):
        """Get default configuration for podcast processing."""
        # Since we're now running from project root, use relative paths
        return {
            # Input/Output paths
            "input_video": "podcast.mp4",
            "base_output_dir": "output",
            
            # Module configuration
            "modules": {
                "clipper": {
                    "enabled": True,
                    "min_clip_duration": 45.0,
                    "max_clip_duration": 90.0,
                    "max_total_clips": 10,
                    "scout_model": "deepseek-r1-distill-qwen-32b",
                    "editor_model": "google/gemma-3-27b",
                    "lm_studio_url": "http://localhost:1234/v1"
                },
                "cropper": {
                    "enabled": True,
                    "ratio": "9:16",
                    "quality": "balanced",
                    "crf": None,
                    "preset": None,
                    "plan_only": False,
                    "frame_skip": 0,
                    "downscale": 0,
                    "encoder": "auto"
                },
                "subs": {
                    "enabled": True,
                    "template": "templates/hype",
                    "vertical_align_offset": 0.70,
                    "max_width_ratio": 0.9,
                    "max_lines": 1
                }
            },
            
            # Processing options
            "cleanup_temp": True,
            "preserve_intermediate": False
        }
    
    def setup_directories(self):
        """Create necessary directories for processing."""
        base_dir = Path(self.config["base_output_dir"])
        
        # Create main output directory
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directories for each module
        self.clipper_output_dir = base_dir / "01_clips"
        self.cropper_output_dir = base_dir / "02_cropped"
        self.subs_output_dir = base_dir / "03_final"
        
        # Create directories
        for dir_path in [self.clipper_output_dir, self.cropper_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.temp_dirs.append(str(dir_path))
        
        # Create final output directory separately (don't add to temp_dirs for cleanup)
        self.subs_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Setup directories:")
        print(f"   Input: {self.config['input_video']}")
        print(f"   Clips: {self.clipper_output_dir}")
        print(f"   Cropped: {self.cropper_output_dir}")
        print(f"   Final: {self.subs_output_dir}")
    
    def run_clipper_module(self):
        """Run the ClipperM module to extract viral clips."""
        if not self.config["modules"]["clipper"]["enabled"]:
            print("⏭️  Clipper module disabled, skipping...")
            return []
        
        print("\n" + "="*60)
        print("🎬 STEP 1: RUNNING CLIPPER MODULE")
        print("="*60)
        
        # Ensure LOCAL_BIN_DIR is available to ClipperM module
        original_path = os.environ.get("PATH", "")
        if LOCAL_BIN_DIR not in original_path:
            os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
            print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for ClipperM module")
        
        # Get clipper configuration
        clipper_config = self.config["modules"]["clipper"]
        
        try:
            # Run clipper main function with configuration
            clips = clipper_main(
                input_video_path=self.config["input_video"],
                output_dir=str(self.clipper_output_dir),
                lm_studio_url=clipper_config["lm_studio_url"],
                scout_model=clipper_config["scout_model"],
                editor_model=clipper_config["editor_model"],
                min_clip_duration=clipper_config["min_clip_duration"],
                max_clip_duration=clipper_config["max_clip_duration"],
                max_total_clips=clipper_config["max_total_clips"]
            )
            
            print(f"✅ Clipper module completed. Generated {len(clips)} clips.")
            return clips
            
        except Exception as e:
            print(f"❌ Clipper module failed: {e}")
            return []
    
        
    def run_cropper_module(self, input_clips):
        """Run the CropperM module on all clips from the previous step."""
        if not self.config["modules"]["cropper"]["enabled"]:
            print("⏭️  Cropper module disabled, skipping...")
            return []
        
        if not input_clips:
            print("❌ No clips to crop. Skipping cropper module.")
            return []
        
        print("\n" + "="*60)
        print("✂️  STEP 2: RUNNING CROPPER MODULE")
        print("="*60)
        
        # Ensure LOCAL_BIN_DIR is available to CropperM module
        original_path = os.environ.get("PATH", "")
        if LOCAL_BIN_DIR not in original_path:
            os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
            print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for CropperM module")
        
        cropper_config = self.config["modules"]["cropper"]
        cropped_clips = []
        
        for i, clip in enumerate(input_clips):
            input_path = clip["file_path"]
            output_path = os.path.join(self.cropper_output_dir, f"cropped_{clip['file_name']}")
            
            print(f"\n🎥 Processing clip {i+1}/{len(input_clips)}: {clip['file_name']}")
            
            try:
                # Run cropper main function
                cropper_main(
                    input_video_path=input_path,
                    output_video_path=output_path,
                    ratio=cropper_config["ratio"],
                    quality=cropper_config["quality"],
                    crf=cropper_config["crf"],
                    preset=cropper_config["preset"],
                    plan_only=cropper_config["plan_only"],
                    frame_skip=cropper_config["frame_skip"],
                    downscale=cropper_config["downscale"],
                    encoder=cropper_config["encoder"]
                )
                
                if os.path.exists(output_path):
                    cropped_clips.append({
                        "file_name": f"cropped_{clip['file_name']}",
                        "file_path": output_path,
                        "title": clip["title"],
                        "original_clip": clip
                    })
                    print(f"✅ Successfully cropped: {output_path}")
                else:
                    print(f"❌ Cropping failed: {output_path}")
                    
            except Exception as e:
                print(f"❌ Error cropping {clip['file_name']}: {e}")
        
        print(f"\n✅ Cropper module completed. Processed {len(cropped_clips)} clips.")
        return cropped_clips
    
    def run_subs_module(self, input_clips):
        """Run the SubsM module on all cropped clips."""
        if not self.config["modules"]["subs"]["enabled"]:
            print("⏭️  Subtitles module disabled, skipping...")
            return []
        
        if not input_clips:
            print("❌ No clips to add subtitles to. Skipping subs module.")
            return []
        
        print("\n" + "="*60)
        print("📝 STEP 3: RUNNING SUBTITLES MODULE")
        print("="*60)
        
        subs_config = self.config["modules"]["subs"]
        final_clips = []
        
        # Ensure LOCAL_BIN_DIR is available to SubsM module
        original_path = os.environ.get("PATH", "")
        if LOCAL_BIN_DIR not in original_path:
            os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
            print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for SubsM module")
        
        try:
            for i, clip in enumerate(input_clips):
                input_path = clip["file_path"]
                output_path = os.path.join(self.subs_output_dir, f"final_{clip['file_name']}")
                
                print(f"\n🎬 Adding subtitles to clip {i+1}/{len(input_clips)}: {clip['file_name']}")
                
                # Run subs main function with configuration
                success = subs_main(
                    input_video_path=input_path,
                    output_video_path=output_path,
                    template_path=subs_config["template"],
                    vertical_align_offset=subs_config["vertical_align_offset"],
                    max_width_ratio=subs_config["max_width_ratio"],
                    max_lines=subs_config["max_lines"]
                )
                
                if success and os.path.exists(output_path):
                    final_clips.append({
                        "file_name": f"final_{clip['file_name']}",
                        "file_path": output_path,
                        "title": clip["title"],
                        "processing_chain": clip
                    })
                    print(f"✅ Successfully added subtitles: {output_path}")
                else:
                    print(f"❌ Subtitle addition failed: {output_path}")
                    
        except Exception as e:
            print(f"❌ Error adding subtitles to {clip['file_name']}: {e}")
        
        print(f"\n✅ Subtitles module completed. Processed {len(final_clips)} clips.")
        return final_clips
    
        
    def cleanup(self):
        """Clean up temporary directories if configured."""
        if not self.config["cleanup_temp"]:
            return
        
        print("\n🧹 Cleaning up temporary files...")
        
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir) and not self.config["preserve_intermediate"]:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"   Removed: {temp_dir}")
                except Exception as e:
                    print(f"   Failed to remove {temp_dir}: {e}")
    
    def run_complete_pipeline(self):
        """Run the complete podcast processing pipeline."""
        print("🚀 Starting Podcast Video Processing Pipeline")
        print("="*60)
        
        try:
            # Setup
            self.setup_directories()
            
            # Step 1: ClipperM - Extract viral clips
            clips = self.run_clipper_module()
            
            # Step 2: CropperM - Crop clips to vertical format
            cropped_clips = self.run_cropper_module(clips)
            
            # Step 3: SubsM - Add subtitles
            final_clips = self.run_subs_module(cropped_clips)
            
            # Summary
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📊 Summary:")
            print(f"   Input video: {self.config['input_video']}")
            print(f"   Clips extracted: {len(clips)}")
            print(f"   Clips cropped: {len(cropped_clips)}")
            print(f"   Final videos: {len(final_clips)}")
            print(f"   Output directory: {self.subs_output_dir}")
            
            if final_clips:
                print(f"\n📹 Final videos:")
                for clip in final_clips:
                    print(f"   - {clip['file_name']}: {clip['title']}")
            
            return final_clips
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            return []
        finally:
            self.cleanup()
            # print("removed cleanup")

def main(config_file=None):
    """
    Main function to run the podcast scenario.
    
    Args:
        config_file (str): Path to JSON configuration file. If None, uses default config.
    """
    # Load configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded configuration from: {config_file}")
    else:
        config = None
        print("📋 Using default configuration")
    
    # Create and run scenario
    scenario = PodcastScenario(config)
    return scenario.run_complete_pipeline()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Podcast Video Processing Pipeline")
    parser.add_argument('--config', type=str, help="Path to JSON configuration file")
    parser.add_argument('--input', type=str, help="Input video file path")
    parser.add_argument('--output', type=str, help="Output directory path")
    parser.add_argument('--clips', type=int, default=10, help="Maximum number of clips to generate")
    
    args = parser.parse_args()
    
    # Load or create configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = PodcastScenario().get_default_config()
        
        # Override with command line arguments
        if args.input:
            config["input_video"] = args.input
        if args.output:
            config["base_output_dir"] = args.output
        config["modules"]["clipper"]["max_total_clips"] = args.clips
    
    # Run scenario
    scenario = PodcastScenario(config)
    final_clips = scenario.run_complete_pipeline()
    
    print(f"\n✅ Processing complete! Generated {len(final_clips)} final videos.")
