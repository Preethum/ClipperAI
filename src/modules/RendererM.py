"""
RendererM — Single-pass video compositor.

Reads a clips manifest and renders all final clips:
  1. Cuts segment from source video
  2. Applies crop/reframe (if "crop" key present)
  3. Burns subtitles via PyCaps (if "subs" key present)
"""

import os
import sys
import json
import subprocess
import shutil
import time

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_bin_dir, setup_bin_path, get_templates_dir
except ImportError:
    def get_project_root():
        # src/modules/RendererM.py -> src/modules -> src -> root
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    def get_bin_dir():
        return os.path.join(get_project_root(), 'bin')
    def get_templates_dir():
        return os.path.join(get_project_root(), 'templates')
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


def _get_ffmpeg_path(binary_name):
    if os.name == 'nt' and not binary_name.endswith('.exe'):
        binary_name = f"{binary_name}.exe"
    
    # Try the local bin directory first
    project_root = get_project_root()
    bin_path = os.path.join(project_root, 'bin', binary_name)
    if os.path.exists(bin_path):
        return bin_path
        
    # Fallback to system path
    return binary_name


def _detect_available_encoders():
    """Detect available video encoders on the system, prioritizing AV1."""
    ffmpeg_path = _get_ffmpeg_path('ffmpeg')
    cmd = [ffmpeg_path, '-encoders']
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        encoders_output = result.stdout
        
        # Priority order: AV1 encoders first, then H.264
        av1_encoders = []
        h264_encoders = []
        
        # Check for AV1 encoders in priority order
        if 'av1_nvenc' in encoders_output:
            av1_encoders.append('av1_nvenc')
        if 'libsvtav1' in encoders_output:
            av1_encoders.append('libsvtav1')
        if 'libaom-av1' in encoders_output:
            av1_encoders.append('libaom-av1')
        if 'librav1e' in encoders_output:
            av1_encoders.append('librav1e')
            
        # Check for H.264 encoders
        if 'h264_nvenc' in encoders_output:
            h264_encoders.append('h264_nvenc')
        if 'libx264' in encoders_output:
            h264_encoders.append('libx264')
            
        return {
            'available_av1': av1_encoders,
            'available_h264': h264_encoders,
            'preferred_av1': av1_encoders[0] if av1_encoders else None,
            'preferred_h264': h264_encoders[0] if h264_encoders else None
        }
        
    except Exception as e:
        print(f"Warning: Could not detect encoders: {e}")
        # Fallback to assuming basic encoders
        return {
            'available_av1': [],
            'available_h264': ['libx264'],
            'preferred_av1': None,
            'preferred_h264': 'libx264'
        }


def _cut_segment(source_video, start, duration, output_path, encoder_config=None):
    # Accurate cut with re-encoding to ensure audio-video sync.
    # We use -ss AFTER -i for slower but frame-accurate seeking.
    # Supports both AV1 and H.264 encoders based on availability.
    
    if encoder_config is None:
        encoder_config = {}
    
    # Detect available encoders
    encoders = _detect_available_encoders()
    
    # Choose encoder based on config or availability
    use_av1 = encoder_config.get('use_av1', True) and encoders['preferred_av1']
    
    if use_av1:
        video_encoder = encoders['preferred_av1']
        # AV1 encoder settings
        if video_encoder == 'libsvtav1':
            video_args = ['-c:v', 'libsvtav1', '-preset', '6', '-crf', '30']
            audio_encoder = 'libopus'
            audio_args = ['-c:a', 'libopus', '-b:a', '128k']
        elif video_encoder == 'libaom-av1':
            video_args = ['-c:v', 'libaom-av1', '-cpu-used', '4', '-crf', '30']
            audio_encoder = 'libopus'
            audio_args = ['-c:a', 'libopus', '-b:a', '128k']
        elif video_encoder == 'av1_nvenc':
            video_args = ['-c:v', 'av1_nvenc', '-preset', 'fast', '-cq', '30']
            audio_encoder = 'aac'
            audio_args = ['-c:a', 'aac', '-b:a', '128k']
        else:
            # Fallback to H.264 if AV1 encoder not recognized
            video_encoder = encoders['preferred_h264']
            video_args = ['-c:v', video_encoder, '-preset', 'ultrafast', '-crf', '17']
            audio_encoder = 'aac'
            audio_args = ['-c:a', 'aac']
        
        print(f"     🎬 Using AV1 encoder: {video_encoder}")
    else:
        # Use H.264
        video_encoder = encoders['preferred_h264']
        video_args = ['-c:v', video_encoder, '-preset', 'ultrafast', '-crf', '17']
        audio_encoder = 'aac'
        audio_args = ['-c:a', 'aac']
        print(f"     🎬 Using H.264 encoder: {video_encoder}")
    
    cmd = [
        _get_ffmpeg_path('ffmpeg'), '-y',
        '-ss', str(start),
        '-i', source_video,
        '-t', str(duration),
    ] + video_args + audio_args + [
        '-avoid_negative_ts', '1',
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg cut failed: {result.stderr.decode()[-2000:]}")


def _apply_crop(input_path, output_path, crop_data):
    sys.path.append(os.path.dirname(__file__))
    from CropperM import main as cropper_main
    cropper_main(
        input_video_path=input_path,
        output_video_path=output_path,
        ratio=crop_data.get("ratio", "9:16"),
        quality=crop_data.get("quality", "balanced"),
        crf=None, preset=None, plan_only=False,
        frame_skip=0, downscale=0,
        encoder=crop_data.get("encoder", "auto")
    )


def _apply_subs(input_path, output_path, subs_data):
    sys.path.append(os.path.dirname(__file__))
    from SubsM import main as subs_main
    success = subs_main(
        input_video_path=input_path,
        output_video_path=output_path,
        template_path=subs_data.get("template", "templates/default"),
        vertical_align_offset=subs_data.get("vertical_align_offset", 0.70),
        max_width_ratio=subs_data.get("max_width_ratio", 0.9),
        max_lines=subs_data.get("max_lines", 1)
    )
    if not success:
        raise RuntimeError("SubsM rendering failed")


def _apply_final_compression(input_path, output_path, encoder_config=None):
    """Apply final compression to reduce file size using AV1 or H.264."""
    if encoder_config is None:
        encoder_config = {}
    
    # Skip compression if disabled
    if not encoder_config.get('final_compression', True):
        shutil.copy2(input_path, output_path)
        return
    
    # Detect available encoders
    encoders = _detect_available_encoders()
    
    # Choose encoder based on config or availability
    use_av1 = encoder_config.get('use_av1', True) and encoders['preferred_av1']
    
    if use_av1:
        video_encoder = encoders['preferred_av1']
        # AV1 encoder settings for final compression (higher quality)
        if video_encoder == 'libsvtav1':
            video_args = ['-c:v', 'libsvtav1', '-preset', '4', '-crf', '25']
            audio_encoder = 'libopus'
            audio_args = ['-c:a', 'libopus', '-b:a', '128k']
        elif video_encoder == 'libaom-av1':
            video_args = ['-c:v', 'libaom-av1', '-cpu-used', '3', '-crf', '25']
            audio_encoder = 'libopus'
            audio_args = ['-c:a', 'libopus', '-b:a', '128k']
        elif video_encoder == 'av1_nvenc':
            video_args = ['-c:v', 'av1_nvenc', '-preset', 'fast', '-cq', '25']
            audio_encoder = 'aac'
            audio_args = ['-c:a', 'aac', '-b:a', '128k']
        else:
            # Fallback to H.264
            video_encoder = encoders['preferred_h264']
            video_args = ['-c:v', video_encoder, '-preset', 'slow', '-crf', '23']
            audio_encoder = 'aac'
            audio_args = ['-c:a', 'aac', '-b:a', '128k']
        
        print(f"     🗜️  Final compression with AV1: {video_encoder}")
    else:
        # Use H.264 for final compression
        video_encoder = encoders['preferred_h264']
        video_args = ['-c:v', video_encoder, '-preset', 'slow', '-crf', '23']
        audio_encoder = 'aac'
        audio_args = ['-c:a', 'aac', '-b:a', '128k']
        print(f"     🗜️  Final compression with H.264: {video_encoder}")
    
    cmd = [
        _get_ffmpeg_path('ffmpeg'), '-y',
        '-i', input_path,
    ] + video_args + audio_args + [
        '-movflags', '+faststart',  # Optimize for web streaming
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Final compression failed: {result.stderr.decode()[-2000:]}")
    
    # Show compression ratio
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    ratio = (1 - compressed_size / original_size) * 100
    print(f"     📊 Size reduction: {ratio:.1f}% ({original_size/1024/1024:.1f}MB → {compressed_size/1024/1024:.1f}MB)")


def main(source_video, manifest, output_dir, config=None):
    """Render all clips from the manifest. Returns list of exported clip metadata."""
    if config is None:
        config = {}
    if not manifest:
        return []

    os.makedirs(output_dir, exist_ok=True)
    exported_clips = []

    # Extract encoder configuration
    encoder_config = config.get('encoder', {})
    if isinstance(encoder_config, str):
        encoder_config = {}
    
    # Show detected encoders
    encoders = _detect_available_encoders()
    if encoders['preferred_av1']:
        print(f"🎬 AV1 encoders available: {', '.join(encoders['available_av1'])}")
        print(f"✅ Using AV1 encoder: {encoders['preferred_av1']}")
    else:
        print(f"⚠️  No AV1 encoders found, using H.264: {encoders['preferred_h264']}")

    import threading
    import concurrent.futures
    list_lock = threading.Lock()
    
    def _render_single_clip(i, clip):
        clip_start = clip["start"]
        clip_end = clip["end"]
        clip_duration = clip_end - clip_start
        file_name = clip.get("file_name", f"clip_{i+1}.mp4")
        final_output = os.path.join(output_dir, file_name)

        has_crop = "crop" in clip
        has_subs = "subs" in clip
        tags = []
        if has_crop: tags.append("crop")
        if has_subs: tags.append("subs")
        tag_str = f" [{'+'.join(tags)}]" if tags else ""

        print(f"     [{i+1}/{len(manifest)}] {file_name} ({clip_duration:.0f}s){tag_str}")

        base_name = os.path.splitext(file_name)[0]
        # Unique temp paths per thread using base_name and i index to avoid collissions
        temp_cut = os.path.join(output_dir, f"_temp_cut_{base_name}_{i}.mp4")
        temp_crop = os.path.join(output_dir, f"_temp_crop_{base_name}_{i}.mp4")
        temp_before_compression = os.path.join(output_dir, f"_temp_pre_compress_{base_name}_{i}.mp4")

        try:
            # CUT (with encoder support)
            _cut_segment(source_video, clip_start, clip_duration, temp_cut, encoder_config)
            current_input = temp_cut

            # CROP
            if has_crop:
                _apply_crop(current_input, temp_crop, clip["crop"])
                if os.path.exists(temp_cut): os.remove(temp_cut)
                current_input = temp_crop

            # SUBS or prepare for compression
            if has_subs:
                _apply_subs(current_input, temp_before_compression, clip["subs"])
                if os.path.exists(current_input) and current_input != temp_before_compression:
                    os.remove(current_input)
            else:
                if current_input != temp_before_compression:
                    shutil.move(current_input, temp_before_compression)

            # FINAL COMPRESSION
            _apply_final_compression(temp_before_compression, final_output, encoder_config)
            
            # Clean up temp files
            if os.path.exists(temp_before_compression):
                os.remove(temp_before_compression)

            if os.path.exists(final_output):
                with list_lock:
                    exported_clips.append({
                        "file_name": file_name,
                        "file_path": final_output,
                        "title": clip.get("title", "Untitled"),
                        "start_time": clip_start,
                        "end_time": clip_end,
                        "duration": clip_duration,
                        "viral_archetype": clip.get("viral_archetype", ""),
                        "engagement_trigger": clip.get("engagement_trigger", ""),
                        "scores": clip.get("scores", {})
                    })

        except Exception as e:
            print(f"     ❌ {file_name}: {e}")
            for tmp in (temp_cut, temp_crop, temp_before_compression):
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except OSError: pass

    # Run in parallel using ThreadPoolExecutor (Safe for independent FFmpeg subprocesses)
    max_workers = min(3, len(manifest))
    if max_workers > 1:
        print(f"     🚀 Processing with {max_workers} parallel GPU workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_render_single_clip, i, clip) for i, clip in enumerate(manifest)]
            concurrent.futures.wait(futures)
    else:
        for i, clip in enumerate(manifest):
            _render_single_clip(i, clip)

    # Save metadata
    if exported_clips:
        metadata_path = os.path.join(output_dir, "clips_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(exported_clips, f, indent=4)

    return exported_clips


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render clips from a manifest file")
    parser.add_argument('-i', '--input', required=True, help="Source video file")
    parser.add_argument('-m', '--manifest', required=True, help="Path to clips_manifest.json")
    parser.add_argument('-o', '--output', required=True, help="Output directory")
    parser.add_argument('--use-av1', action='store_true', default=True, help="Use AV1 encoder if available (default: True)")
    parser.add_argument('--no-av1', action='store_true', help="Force use H.264 encoder instead of AV1")
    parser.add_argument('--no-compression', action='store_true', help="Skip final compression step")
    args = parser.parse_args()
    
    # Build encoder configuration
    encoder_config = {
        'use_av1': args.use_av1 and not args.no_av1,
        'final_compression': not args.no_compression
    }
    
    config = {
        'encoder': encoder_config
    }
    
    with open(args.manifest, 'r') as f:
        manifest_data = json.load(f)
    main(args.input, manifest_data, args.output, config)
