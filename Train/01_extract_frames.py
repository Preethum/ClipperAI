import os
import sys
import subprocess
import argparse

# Add src to sys.path to import path_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from utils.path_utils import get_project_root, setup_bin_path
except ImportError:
    print("Error: Could not import utils.path_utils. Make sure you are running from the project.")
    sys.exit(1)

def _get_ffmpeg_path(binary_name='ffmpeg'):
    if os.name == 'nt' and not binary_name.endswith('.exe'):
        binary_name = f"{binary_name}.exe"
    
    project_root = get_project_root()
    bin_path = os.path.join(project_root, 'bin', binary_name)
    if os.path.exists(bin_path):
        return bin_path
        
    return binary_name

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    # Output pattern ensures files are nicely numbered
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    ffmpeg_path = _get_ffmpeg_path()
    setup_bin_path() # Optional: adds bin to PATH if needed
    
    # FFmpeg command
    command = [
        ffmpeg_path,
        "-err_detect", "ignore_err",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_pattern
    ]
    
    print(f"Extracting frames from {video_path} at {fps} fps to {output_dir}...")
    try:
        subprocess.run(command, check=True)
        print("Done extraction!")
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not found in system PATH.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg process failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an input video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out", type=str, default="extracted_frames", help="Output directory to save frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    args = parser.parse_args()
    
    extract_frames(args.video, args.out, args.fps)
