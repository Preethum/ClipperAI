"""
StitcherM.py - High-quality video stitching utility.
Concatenates multiple video files from a directory into a single output.
"""

import os
import sys
import subprocess
import time
from typing import List

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_bin_dir, setup_bin_path
except ImportError:
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    def get_bin_dir():
        return os.path.join(get_project_root(), 'bin')
    def setup_bin_path():
        bin_dir = get_bin_dir()
        if os.path.exists(bin_dir):
            p = os.environ.get("PATH", "")
            if bin_dir not in p:
                os.environ["PATH"] = bin_dir + os.pathsep + p
            return bin_dir
        return None

LOCAL_BIN_DIR = setup_bin_path()
if LOCAL_BIN_DIR is None:
    LOCAL_BIN_DIR = get_bin_dir()

def get_ffmpeg_path(binary_name):
    """Get the full path to an FFmpeg binary."""
    if os.name == 'nt' and not binary_name.endswith('.exe'):
        binary_name = f"{binary_name}.exe"
    
    project_root = get_project_root()
    bin_path = os.path.join(project_root, 'bin', binary_name)
    if os.path.exists(bin_path):
        return bin_path
        
    return binary_name

def get_video_files(directory: str) -> List[str]:
    """Get all video files in a directory, sorted by name."""
    extensions = ('.mp4', '.mkv', '.mov', '.avi', '.ts')
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]
    # Sort files naturally
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    return sorted(files, key=natural_sort_key)

def detect_hw_encoder():
    """Probes FFmpeg for available hardware H.264 encoders."""
    candidates = [
        ('h264_nvenc',        'nvenc'),
        ('h264_videotoolbox', 'videotoolbox'),
    ]
    for encoder, etype in candidates:
        try:
            result = subprocess.run(
                [get_ffmpeg_path('ffmpeg'), '-hide_banner', '-encoders'],
                capture_output=True, text=True
            )
            if encoder in result.stdout:
                return encoder, etype
        except Exception:
            break
    return 'libx264', 'libx264'

def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file using ffprobe."""
    try:
        cmd = [
            get_ffmpeg_path('ffprobe'), '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0

def run_ffmpeg_with_progress(command, total_duration, desc="Stitching"):
    """Runs an FFmpeg command and shows a tqdm progress bar based on stderr output."""
    from tqdm import tqdm
    import re
    
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    pbar = tqdm(total=int(total_duration), desc=desc, unit="s", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]')
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+)\.(\d+)')
    last_seconds = 0
    stderr_lines = []
    
    try:
        for line in process.stderr:
            stderr_lines.append(line)
            match = time_pattern.search(line)
            if match:
                h, m, s, _ = match.groups()
                current_seconds = int(h) * 3600 + int(m) * 60 + int(s)
                if current_seconds > last_seconds:
                    pbar.update(current_seconds - last_seconds)
                    last_seconds = current_seconds
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    
    # Ensure progress bar reaches 100%
    if total_duration > last_seconds:
        pbar.update(int(total_duration) - last_seconds)
    
    pbar.close()
    process.wait()
    return process.returncode, ''.join(stderr_lines)

def stitch_videos(input_dir: str, output_path: str, high_quality: bool = True, fast_mode: bool = False):
    """
    Stitches videos in input_dir into output_path.
    
    - fast_mode: Uses '-c copy' if True (requires identical resolutions/codecs).
    - high_quality: Uses slow encoding with CRF 18 if True.
    """
    video_files = get_video_files(input_dir)
    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        return False

    print(f"🎬 Found {len(video_files)} videos to stitch.")
    
    # Calculate total duration for progress bar
    print("⏳ Calculating total duration...")
    total_duration = sum(get_video_duration(f) for f in video_files)
    if total_duration > 0:
        print(f"🕒 Total estimated duration: {total_duration:.2f}s")
    
    # Create the concat file for FFmpeg
    concat_file_path = "concat_list.txt"
    try:
        with open(concat_file_path, "w", encoding="utf-8") as f:
            for video in video_files:
                # FFmpeg concat demuxer requires escaped paths
                abs_path = os.path.abspath(video).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        ffmpeg_cmd = [
            get_ffmpeg_path('ffmpeg'), '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file_path
        ]

        if fast_mode:
            print("🚀 Fast mode enabled: Copying streams (no re-encoding).")
            ffmpeg_cmd += ['-c', 'copy']
        else:
            encoder_name, encoder_type = detect_hw_encoder()
            print(f"💎 {'High quality' if high_quality else 'Balanced'} mode: Re-encoding using {encoder_name}.")
            
            ffmpeg_cmd += ['-c:v', encoder_name]
            
            if encoder_type == 'nvenc':
                ffmpeg_cmd += ['-cq', '18' if high_quality else '23', '-preset', 'p7' if high_quality else 'p4']
            elif encoder_type == 'videotoolbox':
                ffmpeg_cmd += ['-q:v', '75' if high_quality else '50'] # Placeholder for quality
            else:
                # libx264 software fallback
                ffmpeg_cmd += [
                    '-crf', '18' if high_quality else '23',
                    '-preset', 'medium' # 'slow' is too slow for most users, 'medium' is a better default
                ]

            ffmpeg_cmd += [
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-vsync', 'vfr'
            ]

        ffmpeg_cmd.append(output_path)

        print(f"🔧 Starting FFmpeg process...")
        
        if total_duration > 0:
            returncode, stderr = run_ffmpeg_with_progress(ffmpeg_cmd, total_duration)
        else:
            # Fallback if duration couldn't be determined
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            returncode, stderr = result.returncode, result.stderr

        if returncode == 0:
            print(f"✅ Success! Stitched video saved to: {output_path}")
            return True
        else:
            print(f"❌ FFmpeg failed with error:\n{stderr}")
            return False

    finally:
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stitch multiple videos from a folder into one.")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing videos")
    parser.add_argument("--output", "-o", required=True, help="Output video file path")
    parser.add_argument("--fast", action="store_true", help="Fast mode: use -c copy (no re-encoding, requires identical files)")
    parser.add_argument("--balanced", action="store_true", help="Lower quality (CRF 23) but faster encoding")

    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = stitch_videos(
        input_dir=args.input,
        output_path=args.output,
        high_quality=not args.balanced,
        fast_mode=args.fast
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
