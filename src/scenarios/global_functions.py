"""
Global functions to interface with ClipperM, CropperM, and SubsM modules.
This file contains all the core functionality that can be used by different scenarios.
"""

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

def setup_directories(base_output_dir):
    """
    Create necessary directories for processing.
    
    Args:
        base_output_dir (str): Base output directory path
        
    Returns:
        dict: Dictionary containing directory paths
    """
    base_dir = Path(base_output_dir)
    
    # Create main output directory
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directories for each module
    clipper_output_dir = base_dir / "01_clips"
    cropper_output_dir = base_dir / "02_cropped"
    # Final output goes directly to the base directory
    subs_output_dir = base_dir
    
    # Create directories
    temp_dirs = []
    for dir_path in [clipper_output_dir, cropper_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        temp_dirs.append(str(dir_path))
    
    directories = {
        "clipper_output_dir": clipper_output_dir,
        "cropper_output_dir": cropper_output_dir,
        "subs_output_dir": subs_output_dir,
        "temp_dirs": temp_dirs
    }
    
    print(f"📁 Setup directories:")
    print(f"   Clips: {clipper_output_dir}")
    print(f"   Cropped: {cropper_output_dir}")
    print(f"   Final: {subs_output_dir}")
    
    return directories

def run_clipper_module(input_video, output_dir, config):
    """
    Run ClipperM module to extract viral clips.
    
    Args:
        input_video (str): Path to input video file
        output_dir (str): Directory to save output clips
        config (dict): Clipper configuration
        
    Returns:
        list: List of generated clips
    """
    if not config.get("enabled", True):
        print("⏭️  Clipper module disabled, skipping...")
        return []
    
    print("\n" + "="*60)
    print("🎬 STEP 1: RUNNING CLIPPER MODULE")
    print("="*60)
    print(f"🔧 Scout Model: {config.get('scout_model', 'default')}")
    print(f"🔧 Editor Model: {config.get('editor_model', 'default')}")
    print(f"🔧 Viral Archetypes: {config.get('viral_archetypes', 'Not specified')}")
    
    # Ensure LOCAL_BIN_DIR is available to ClipperM module
    original_path = os.environ.get("PATH", "")
    if LOCAL_BIN_DIR not in original_path:
        os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
        print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for ClipperM module")
    
    try:
        print(f"[DEBUG GLOBAL_FUNCTIONS] Config vision_concurrency is: {config.get('vision_concurrency')}")
        # Run clipper main function - all parameters must be defined in scenario config
        clips = clipper_main(
            input_video_path=input_video,
            output_dir=output_dir,
            lm_studio_url=config["lm_studio_url"],
            scout_model=config["scout_model"],
            editor_model=config["editor_model"],
            min_clip_duration=config["min_clip_duration"],
            max_clip_duration=config["max_clip_duration"],
            max_total_clips=config["max_total_clips"],
            viral_archetypes=config["viral_archetypes"],
            scout_system_instruction=config["scout_system_instruction"],
            scout_user_prompt=config["scout_user_prompt"],
            editor_system_instruction=config["editor_system_instruction"],
            editor_user_prompt=config["editor_user_prompt"],
            deduplication_threshold=config["deduplication_threshold"],
            enable_ocr=config["enable_ocr"],
            enable_vision=config["enable_vision"],
            vision_model=config["vision_model"],
            vision_interval=config["vision_interval"],
            vision_concurrency=config["vision_concurrency"]
        )
        
        print(f"✅ Clipper module completed. Generated {len(clips)} clips.")
        return clips
        
    except Exception as e:
        print(f"❌ Clipper module failed: {e}")
        return []

def run_cropper_module(input_clips, output_dir, config):
    """
    Run the CropperM module on all clips from the previous step.
    
    Args:
        input_clips (list): List of clips from clipper module
        output_dir (str): Directory to save cropped clips
        config (dict): Cropper configuration
        
    Returns:
        list: List of cropped clips
    """
    if not config.get("enabled", True):
        print("⏭️  Cropper module disabled, skipping...")
        return []
    
    if not input_clips:
        print("❌ No clips to crop. Skipping cropper module.")
        return []
    
    print("\n" + "="*60)
    print("✂️  STEP 2: RUNNING CROPPER MODULE")
    print("="*60)
    print(f"🔧 Ratio: {config.get('ratio', '9:16')}")
    print(f"🔧 Quality: {config.get('quality', 'balanced')}")
    
    # Ensure LOCAL_BIN_DIR is available to CropperM module
    original_path = os.environ.get("PATH", "")
    if LOCAL_BIN_DIR not in original_path:
        os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
        print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for CropperM module")
    
    cropped_clips = []
    
    for i, clip in enumerate(input_clips):
        input_path = clip["file_path"]
        output_path = os.path.join(output_dir, clip['file_name'])
        
        print(f"\n🎥 Processing clip {i+1}/{len(input_clips)}: {clip['file_name']}")
        
        try:
            # Run cropper main function with only accepted parameters
            cropper_main(
                input_video_path=input_path,
                output_video_path=output_path,
                ratio=config.get("ratio", "9:16"),
                quality=config.get("quality", "balanced"),
                crf=config.get("crf"),
                preset=config.get("preset"),
                plan_only=config.get("plan_only", False),
                frame_skip=config.get("frame_skip", 0),
                downscale=config.get("downscale", 0),
                encoder=config.get("encoder", "auto")
            )
            
            if os.path.exists(output_path):
                cropped_clips.append({
                    "file_name": clip['file_name'],
                    "file_path": output_path,
                    "title": clip.get('title', 'Untitled'),
                    "start_time": clip.get('start_time', 0),
                    "end_time": clip.get('end_time', 0),
                    "duration": clip.get('duration', 0)
                })
                print(f"✅ Cropped: {output_path}")
            else:
                print(f"❌ Cropping failed: {output_path}")
                
        except Exception as e:
            print(f"❌ Cropping failed for {clip['file_name']}: {e}")
            continue
    
    print(f"\n✅ Cropper module completed. Processed {len(cropped_clips)}/{len(input_clips)} clips.")
    return cropped_clips

def run_subs_module(input_clips, output_dir, config):
    """
    Run the SubsM module on all cropped clips.
    
    Args:
        input_clips (list): List of clips from cropper module
        output_dir (str): Directory to save final clips with subtitles
        config (dict): Subs configuration
        
    Returns:
        list: List of final clips with subtitles
    """
    if not config.get("enabled", True):
        print("⏭️  Subs module disabled, skipping...")
        return []
    
    if not input_clips:
        print("❌ No clips to add subtitles to. Skipping subs module.")
        return []
    
    print("\n" + "="*60)
    print("📝 STEP 3: RUNNING SUBS MODULE")
    print("="*60)
    print(f"🔧 Template: {config.get('template', 'templates/default')}")
    print(f"🔧 Vertical Align: {config.get('vertical_align_offset', 0.70)}")
    print(f"🔧 Max Width Ratio: {config.get('max_width_ratio', 0.9)}")
    print(f"🔧 Max Lines: {config.get('max_lines', 1)}")
    
    # Ensure LOCAL_BIN_DIR is available to SubsM module
    original_path = os.environ.get("PATH", "")
    if LOCAL_BIN_DIR not in original_path:
        os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
        print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for SubsM module")
    
    final_clips = []
    
    for i, clip in enumerate(input_clips):
        input_path = clip["file_path"]
        # Use the same file name (no 'final_' prefix)
        output_path = os.path.join(output_dir, clip['file_name'])
        
        print(f"\n🎬 Adding subtitles to clip {i+1}/{len(input_clips)}: {clip['file_name']}")
        
        # Run subs main function with only accepted parameters
        success = subs_main(
            input_video_path=input_path,
            output_video_path=output_path,
            template_path=config.get("template", "templates/default"),
            vertical_align_offset=config.get("vertical_align_offset", 0.70),
            max_width_ratio=config.get("max_width_ratio", 0.9),
            max_lines=config.get("max_lines", 1)
        )
        
        if success and os.path.exists(output_path):
            final_clips.append({
                "file_name": clip['file_name'],
                "file_path": output_path,
                "title": clip.get('title', 'Untitled'),
                "start_time": clip.get('start_time', 0),
                "end_time": clip.get('end_time', 0),
                "duration": clip.get('duration', 0)
            })
            print(f"✅ Subtitles added: {output_path}")
        else:
            print(f"❌ Subtitle addition failed: {output_path}")
    
    print(f"\n✅ Subs module completed. Processed {len(final_clips)}/{len(input_clips)} clips.")
    return final_clips

def cleanup_temp_directories(temp_dirs, preserve_intermediate=False):
    """
    Clean up temporary directories if configured.
    
    Args:
        temp_dirs (list): List of temporary directory paths
        preserve_intermediate (bool): Whether to preserve intermediate files
    """
    if not temp_dirs:
        return
    
    print("\n🧹 Cleaning up temporary files...")
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir) and not preserve_intermediate:
            try:
                shutil.rmtree(temp_dir)
                print(f"   Removed: {temp_dir}")
            except Exception as e:
                print(f"   Failed to remove {temp_dir}: {e}")

def run_complete_pipeline(config):
    """
    Run the complete processing pipeline using global functions.
    
    Args:
        config (dict): Complete configuration dictionary
        
    Returns:
        list: List of final processed clips
    """
    print("🚀 Starting Video Processing Pipeline")
    print("="*60)
    
    try:
        # Setup directories
        directories = setup_directories(config.get("base_output_dir", "output"))
        
        # Step 1: ClipperM - Extract viral clips
        clips = run_clipper_module(
            config.get("input_video"),
            str(directories["clipper_output_dir"]),
            config.get("modules", {}).get("clipper", {})
        )
        
        # Step 2: CropperM - Crop clips to vertical format
        cropped_clips = run_cropper_module(
            clips,
            str(directories["cropper_output_dir"]),
            config.get("modules", {}).get("cropper", {})
        )
        
        # Step 3: SubsM - Add subtitles
        final_clips = run_subs_module(
            cropped_clips,
            str(directories["subs_output_dir"]),
            config.get("modules", {}).get("subs", {})
        )
        
        # Summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Summary:")
        print(f"   Input video: {config.get('input_video', 'Not specified')}")
        print(f"   Clips extracted: {len(clips)}")
        print(f"   Clips cropped: {len(cropped_clips)}")
        print(f"   Final videos: {len(final_clips)}")
        print(f"   Output directory: {directories['subs_output_dir']}")
        
        if final_clips:
            print(f"\n📹 Final videos:")
            for clip in final_clips:
                print(f"   - {clip.get('file_name', 'unknown')}: {clip.get('title', 'Untitled')}")
            
            # Export final metadata JSON alongside the videos
            metadata_path = os.path.join(str(directories['subs_output_dir']), 'clips_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(final_clips, f, indent=4)
            print(f"\n📋 Metadata saved: {metadata_path}")
        
        return final_clips
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return []
    finally:
        cleanup_temp_directories(
            directories.get("temp_dirs", []),
            config.get("preserve_intermediate", False)
        )
