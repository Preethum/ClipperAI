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
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    if os.name == 'nt':
        binary_name = f"{binary_name}.exe"
    path = os.path.join(LOCAL_BIN_DIR, binary_name)
    return path if os.path.exists(path) else binary_name


def _cut_segment(source_video, start, duration, output_path):
    cmd = [
        _get_ffmpeg_path('ffmpeg'), '-y',
        '-ss', str(start), '-t', str(duration),
        '-i', source_video,
        '-c', 'copy', '-avoid_negative_ts', '1',
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg cut failed: {result.stderr.decode()[:200]}")


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


def main(source_video, manifest, output_dir, config=None):
    """Render all clips from the manifest. Returns list of exported clip metadata."""
    if config is None:
        config = {}
    if not manifest:
        return []

    os.makedirs(output_dir, exist_ok=True)
    exported_clips = []

    for i, clip in enumerate(manifest):
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
        temp_cut = os.path.join(output_dir, f"_temp_cut_{base_name}.mp4")
        temp_crop = os.path.join(output_dir, f"_temp_crop_{base_name}.mp4")

        try:
            # CUT
            _cut_segment(source_video, clip_start, clip_duration, temp_cut)
            current_input = temp_cut

            # CROP
            if has_crop:
                _apply_crop(current_input, temp_crop, clip["crop"])
                if os.path.exists(temp_cut): os.remove(temp_cut)
                current_input = temp_crop

            # SUBS
            if has_subs:
                _apply_subs(current_input, final_output, clip["subs"])
                if os.path.exists(current_input) and current_input != final_output:
                    os.remove(current_input)
            else:
                if current_input != final_output:
                    shutil.move(current_input, final_output)

            if os.path.exists(final_output):
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
            for tmp in (temp_cut, temp_crop):
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except OSError: pass

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
    args = parser.parse_args()
    with open(args.manifest, 'r') as f:
        manifest_data = json.load(f)
    main(args.input, manifest_data, args.output)
