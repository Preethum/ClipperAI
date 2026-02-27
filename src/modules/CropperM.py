import sys
import time
import subprocess
import argparse
import os

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_bin_dir, setup_bin_path
except ImportError:
    # Fallback if utils module not available
    def get_project_root():
        # More robust fallback - try multiple methods
        current_file = os.path.abspath(__file__)
        # We're in src/modules, so go up 2 levels to project root
        return os.path.dirname(os.path.dirname(current_file))
    def get_bin_dir():
        return os.path.join(get_project_root(), 'bin')
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

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root using utility function
PROJECT_ROOT = get_project_root()
# Path to FFmpeg binaries
BIN_DIR = get_bin_dir()

def get_ffmpeg_path(binary_name):
    """Get the full path to an FFmpeg binary from the bin folder."""
    if os.name == 'nt':  # Windows
        binary_name = f"{binary_name}.exe"
    else:  # Linux/Mac
        binary_name = binary_name
    
    binary_path = os.path.join(BIN_DIR, binary_name)
    if os.path.exists(binary_path):
        return binary_path
    else:
        # Fallback to system PATH if not found in bin
        print(f"⚠️  {binary_name} not found in {BIN_DIR}, using system PATH")
        return binary_name

# --- Constants ---
ASPECT_RATIO = 9 / 16

# Lazy-loaded models — initialized on first use so that importing the module
# or running --help doesn't trigger heavyweight model loading.
_model = None
_face_cascade = None

def get_yolo_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')
    return _model

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade

def analyze_scene_content(video_path, scene_start_time, scene_end_time):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    results = get_yolo_model()([frame], verbose=False)

    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_box = [x1, y1, x2, y2]

                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = get_face_cascade().detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                face_box = None
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({'person_box': person_box, 'face_box': face_box})

    cap.release()
    return detected_objects


def analyze_scene_multiframe(video_path, scene_start_sec, scene_end_sec, frame_timestamps):
    """
    Analyze multiple frames within a scene for more robust person detection.
    Uses pre-computed vision frame timestamps instead of just the middle frame.
    Returns averaged detection results.
    """
    import cv2

    # Find all vision timestamps that fall within this scene
    scene_timestamps = [t for t in frame_timestamps if scene_start_sec <= t <= scene_end_sec]
    if not scene_timestamps:
        return []  # Caller should fall back to middle-frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    all_detections = []  # List of per-frame detection lists

    for ts in scene_timestamps:
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        results = get_yolo_model()([frame], verbose=False)
        frame_objects = []
        for result in results:
            for box in result.boxes:
                if box.cls[0] == 0:  # person class
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    person_box = [x1, y1, x2, y2]
                    person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    faces = get_face_cascade().detectMultiScale(
                        person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    face_box = None
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]
                    frame_objects.append({'person_box': person_box, 'face_box': face_box})
        if frame_objects:
            all_detections.append(frame_objects)

    cap.release()

    if not all_detections:
        return []

    # Average across frames: use the frame with the median person count
    # to avoid outlier frames (e.g., transitions with 0 or too many detections)
    all_detections.sort(key=len)
    median_idx = len(all_detections) // 2
    best_detection = all_detections[median_idx]

    # If we have multiple frames to average from, average the bounding boxes
    # of the most common person count
    target_count = len(best_detection)
    matching = [d for d in all_detections if len(d) == target_count]

    if len(matching) <= 1:
        return best_detection

    # Average person boxes across matching frames
    averaged = []
    for person_idx in range(target_count):
        avg_person = [0, 0, 0, 0]
        avg_face = None
        face_count = 0
        face_accum = [0, 0, 0, 0]

        for frame_det in matching:
            pb = frame_det[person_idx]['person_box']
            for j in range(4):
                avg_person[j] += pb[j]
            fb = frame_det[person_idx].get('face_box')
            if fb:
                face_count += 1
                for j in range(4):
                    face_accum[j] += fb[j]

        n = len(matching)
        avg_person = [int(v / n) for v in avg_person]
        if face_count > 0:
            avg_face = [int(v / face_count) for v in face_accum]

        averaged.append({'person_box': avg_person, 'face_box': avg_face})

    return averaged


def detect_scenes(video_path, downscale=0, frame_skip=0):
    """Detect scene boundaries.

    Args:
        video_path: Path to the video file.
        downscale: Downscale factor for processing (0 = auto-detect based on
                   resolution).  Higher values are faster but may miss subtle cuts.
        frame_skip: Number of frames to skip between each processed frame.
                    0 = process every frame (default, most accurate).
    """
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    if downscale > 0:
        video_manager.set_downscale_factor(downscale)
    else:
        video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True,
                                frame_skip=frame_skip)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    return scene_list, fps

def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]

def decide_cropping_strategy(scene_analysis, frame_height):
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * ASPECT_RATIO
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None

def calculate_crop_box(target_box, frame_width, frame_height):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * ASPECT_RATIO)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2

def get_video_properties(video_path):
    """Returns (width, height, fps) from OpenCV — the same backend that reads frames."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def get_media_info(video_path):
    """Returns a dict with human-readable info about the input file."""
    info = {}
    try:
        result = subprocess.run(
            [get_ffmpeg_path('ffprobe'), '-v', 'error', '-show_entries',
             'format=duration,size',
             '-show_entries', 'stream=codec_name,codec_type,width,height,r_frame_rate',
             '-of', 'json', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            fmt = data.get('format', {})
            info['duration'] = float(fmt.get('duration', 0))
            info['size_bytes'] = int(fmt.get('size', 0))
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and 'video_codec' not in info:
                    info['video_codec'] = stream.get('codec_name', 'unknown')
                    info['width'] = stream.get('width', 0)
                    info['height'] = stream.get('height', 0)
                    rate = stream.get('r_frame_rate', '0/1')
                    parts = rate.split('/')
                    if len(parts) == 2 and int(parts[1]) != 0:
                        info['fps'] = round(int(parts[0]) / int(parts[1]), 2)
                    else:
                        info['fps'] = float(parts[0])
                elif stream.get('codec_type') == 'audio' and 'audio_codec' not in info:
                    info['audio_codec'] = stream.get('codec_name', 'unknown')
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return info

def format_duration(seconds):
    """Formats seconds into a human-readable string like '1h 32m 15s'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

def format_file_size(size_bytes):
    """Formats bytes into a human-readable string."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"

def has_audio_stream(video_path):
    """Uses ffprobe to check whether the file contains an audio stream."""
    try:
        result = subprocess.run(
            [get_ffmpeg_path('ffprobe'), '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        return result.returncode == 0 and 'audio' in result.stdout
    except FileNotFoundError:
        # ffprobe not available — assume audio exists and let ffmpeg handle it
        return True

def get_stream_start_time(video_path, stream_type='v:0'):
    """Returns the start_time of a stream in seconds (0.0 if unavailable)."""
    try:
        result = subprocess.run(
            [get_ffmpeg_path('ffprobe'), '-v', 'error', '-select_streams', stream_type,
             '-show_entries', 'stream=start_time', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0.0

def is_variable_frame_rate(video_path):
    """Uses ffprobe to check if the video has a variable frame rate."""
    try:
        result = subprocess.run(
            [get_ffmpeg_path('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        # ffprobe returns "num/den" for both r_frame_rate and avg_frame_rate
        parts = result.stdout.strip().split(',')
        if len(parts) < 2:
            return False
        def parse_rate(s):
            nums = s.strip().split('/')
            if len(nums) == 2 and int(nums[1]) != 0:
                return int(nums[0]) / int(nums[1])
            return float(nums[0])
        r_fps = parse_rate(parts[0])
        avg_fps = parse_rate(parts[1])
        # If the real frame rate and average frame rate differ significantly, it's VFR
        return abs(r_fps - avg_fps) > 0.5
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        return False

def run_ffmpeg_with_progress(command, total_duration, desc="Processing"):
    """Runs an FFmpeg command and shows a tqdm progress bar based on stderr output.
    Returns (returncode, stderr_text) so callers can print errors."""
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
    for line in process.stderr:
        stderr_lines.append(line)
        match = time_pattern.search(line)
        if match:
            h, m, s, _ = match.groups()
            current_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            if current_seconds > last_seconds:
                pbar.update(current_seconds - last_seconds)
                last_seconds = current_seconds
    pbar.update(max(0, int(total_duration) - last_seconds))
    pbar.close()
    process.wait()
    return process.returncode, ''.join(stderr_lines)

def normalize_to_cfr(video_path, output_path, total_duration=0):
    """Re-muxes a VFR video to constant frame rate."""
    print("  Normalizing variable frame rate to constant frame rate...")
    command = [
        get_ffmpeg_path('ffmpeg'), '-y', '-i', video_path,
        '-vsync', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', output_path
    ]
    if total_duration > 0:
        returncode, stderr_text = run_ffmpeg_with_progress(command, total_duration, desc="VFR → CFR")
    else:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Warning: VFR normalization failed, proceeding with original file.")
            print("  Stderr:", e.stderr.decode())
            return False
    if returncode != 0:
        print(f"  Warning: VFR normalization failed, proceeding with original file.")
        return False
    return True

def detect_hw_encoder():
    """Probes FFmpeg for available hardware H.264 encoders.

    Returns (encoder_name, encoder_type) where encoder_type is one of
    'videotoolbox', 'nvenc', or 'libx264'.
    """
    candidates = [
        ('h264_videotoolbox', 'videotoolbox'),
        ('h264_nvenc',        'nvenc'),
    ]
    for encoder, etype in candidates:
        try:
            result = subprocess.run(
                [get_ffmpeg_path('ffmpeg'), '-hide_banner', '-encoders'],
                capture_output=True, text=True
            )
            if encoder in result.stdout:
                return encoder, etype
        except FileNotFoundError:
            break
    return 'libx264', 'libx264'

def resolve_encoder(requested, hw_encoder_name, hw_encoder_type):
    """Resolves which encoder to use based on user request.

    requested: 'auto' (default, always libx264 for quality), 'hw' (use hardware
               if available), or a specific encoder name like 'h264_videotoolbox'.
    Returns (encoder_name, encoder_type).
    """
    if requested == 'auto':
        return 'libx264', 'libx264'
    elif requested == 'hw':
        return hw_encoder_name, hw_encoder_type
    else:
        # User specified an explicit encoder
        if requested == hw_encoder_name:
            return hw_encoder_name, hw_encoder_type
        return requested, requested

def build_encoder_args(encoder_type, quality_level, crf_override=None, preset_override=None):
    """Returns a list of FFmpeg encoder arguments for the given encoder and quality.

    quality_level is one of 'fast', 'balanced', 'high'.
    crf_override and preset_override allow user to force specific values (libx264 only).
    """
    presets = {
        'libx264': {
            'fast':     ['-crf', '28', '-preset', 'veryfast'],
            'balanced': ['-crf', '23', '-preset', 'fast'],
            'high':     ['-crf', '18', '-preset', 'slow'],
        },
        'videotoolbox': {
            'fast':     ['-b:v', '3M', '-allow_sw', '1', '-realtime', '0'],
            'balanced': ['-b:v', '6M', '-allow_sw', '1', '-realtime', '0'],
            'high':     ['-b:v', '12M', '-allow_sw', '1', '-realtime', '0'],
        },
        'nvenc': {
            'fast':     ['-cq', '28', '-preset', 'p1'],
            'balanced': ['-cq', '23', '-preset', 'p4'],
            'high':     ['-cq', '18', '-preset', 'p7'],
        },
    }

    args = list(presets[encoder_type][quality_level])

    # Allow user overrides for libx264
    if encoder_type == 'libx264':
        if crf_override is not None:
            args[args.index('-crf') + 1] = str(crf_override)
        if preset_override is not None:
            args[args.index('-preset') + 1] = preset_override

    return args

def plan(source_video, manifest, config):
    """Enrich manifest with crop planning data for each clip (no video rendering).
    
    Runs scene detection + YOLO analysis on each clip's segment of the source video
    and adds a "crop" key to each manifest entry with per-scene strategy info.
    
    Returns the enriched manifest.
    """
    import cv2

    ratio_str = config.get("ratio", "9:16")
    try:
        rw, rh = ratio_str.split(':')
        aspect_ratio = int(rw) / int(rh)
    except (ValueError, IndexError, ZeroDivisionError):
        print(f"❌ Invalid aspect ratio '{ratio_str}', defaulting to 9:16")
        aspect_ratio = 9 / 16

    frame_skip = config.get("frame_skip", 0)
    downscale = config.get("downscale", 0)

    # Get source video properties once
    orig_w, orig_h, fps = get_video_properties(source_video)
    out_h = orig_h + (orig_h % 2)  # Ensure even
    out_w = int(out_h * aspect_ratio)
    out_w += out_w % 2  # Ensure even

    # Vision frame timestamps for multi-frame YOLO analysis (from ClipperM)
    vision_timestamps = config.get("vision_frame_timestamps", [])

    # Resolve encoder info for manifest (Renderer will use this)
    hw_enc_name, hw_enc_type = detect_hw_encoder()
    enc_name, enc_type = resolve_encoder(config.get("encoder", "auto"), hw_enc_name, hw_enc_type)
    enc_args = build_encoder_args(enc_type, config.get("quality", "balanced"),
                                   crf_override=config.get("crf"), preset_override=config.get("preset"))


    # Run scene detection ONCE on the full source video
    all_scenes, _ = detect_scenes(source_video, downscale=downscale, frame_skip=frame_skip)

    for i, clip in enumerate(manifest):
        clip_start = clip["start"]
        clip_end = clip["end"]
        clip_duration = clip_end - clip_start

        # Filter scenes to this clip's time range
        clip_scenes = [(s, e) for s, e in all_scenes
                       if e.get_seconds() > clip_start and s.get_seconds() < clip_end]

        # Analyze each scene
        scene_plans = []
        track_count = 0
        for start_time, end_time in clip_scenes:
            # Use multi-frame analysis if vision timestamps are available
            if vision_timestamps:
                analysis = analyze_scene_multiframe(
                    source_video,
                    start_time.get_seconds(),
                    end_time.get_seconds(),
                    vision_timestamps
                )
                # Fall back to single middle-frame if no vision frames in this scene
                if not analysis:
                    analysis = analyze_scene_content(source_video, start_time, end_time)
            else:
                analysis = analyze_scene_content(source_video, start_time, end_time)
            strategy, target_box = decide_cropping_strategy(analysis, orig_h)
            
            crop_box = None
            if strategy == 'TRACK' and target_box:
                crop_box = list(calculate_crop_box(target_box, orig_w, orig_h))
                track_count += 1

            scene_plans.append({
                "start_seconds": start_time.get_seconds(),
                "end_seconds": end_time.get_seconds(),
                "start_frame": start_time.get_frames(),
                "end_frame": end_time.get_frames(),
                "strategy": strategy,
                "target_box": target_box,
                "crop_box": crop_box,
                "num_people": len(analysis)
            })

        clip["crop"] = {
            "enabled": True,
            "ratio": ratio_str,
            "aspect_ratio": aspect_ratio,
            "output_width": out_w,
            "output_height": out_h,
            "source_width": orig_w,
            "source_height": orig_h,
            "fps": fps,
            "encoder": enc_name,
            "encoder_type": enc_type,
            "encoder_args": enc_args,
            "quality": config.get("quality", "balanced"),
            "scenes": scene_plans
        }

        lb_count = len(scene_plans) - track_count
        print(f"     [{i+1}/{len(manifest)}] {clip.get('title','?')[:40]}  ({len(scene_plans)} scenes: {track_count}T/{lb_count}L)")

    return manifest

def main(input_video_path, output_video_path, ratio='9:16', quality='balanced', crf=None, preset=None, 
         plan_only=False, frame_skip=0, downscale=0, encoder='auto'):
    """
    Smartly crops a horizontal video into a vertical one.
    
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to the output video file.
        ratio (str): Output aspect ratio as W:H (default: '9:16'). Examples: 9:16, 4:5, 1:1
        quality (str): Encoding quality preset (default: 'balanced'). fast=quick encode, balanced=good quality, high=best quality/slow
        crf (int): Override CRF value directly (0-51, lower=better quality). Overrides quality.
        preset (str): Override FFmpeg x264 preset directly (ultrafast..veryslow). Overrides quality.
        plan_only (bool): Only run scene detection and analysis (Steps 1-3), then print the processing plan without encoding.
        frame_skip (int): Frames to skip during scene detection (default: 0 = every frame, most accurate).
        downscale (int): Downscale factor for scene detection (default: 0 = auto).
        encoder (str): Video encoder: 'auto' (libx264, default), 'hw' (auto-detect hardware encoder), or a specific encoder name.
    """

    # Parse aspect ratio
    try:
        ratio_parts = ratio.split(':')
        ASPECT_RATIO = int(ratio_parts[0]) / int(ratio_parts[1])
    except (ValueError, IndexError, ZeroDivisionError):
        print(f"❌ Invalid aspect ratio '{ratio}'. Use format W:H (e.g. 9:16, 4:5, 1:1)")
        sys.exit(1)

    # Resolve encoder: default is libx264 for best quality; --encoder hw for hardware
    hw_encoder_name, hw_encoder_type = detect_hw_encoder()
    encoder_name, encoder_type = resolve_encoder(encoder, hw_encoder_name, hw_encoder_type)
    enc_args = build_encoder_args(encoder_type, quality,
                                  crf_override=crf, preset_override=preset)

    # Defer heavy imports until after arg parsing so --help is instant
    import cv2
    import numpy as np
    from tqdm import tqdm

    script_start_time = time.time()

    input_video = input_video_path
    final_output_video = output_video_path

    # Ensure the output filename has a video extension so FFmpeg can determine the format
    _, ext = os.path.splitext(final_output_video)
    if not ext:
        final_output_video += '.mp4'
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.mkv"
    temp_cfr_input = f"{base_name}_temp_cfr_input.mp4"
    
    def cleanup_temp_files():
        """Remove any leftover temporary files."""
        for f in [temp_video_output, temp_audio_output, temp_cfr_input]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # Clean up previous temp files if they exist
    cleanup_temp_files()
    if os.path.exists(final_output_video): os.remove(final_output_video)

    # Print input file summary
    media_info = get_media_info(input_video)
    if media_info:
        print(f"\n📄 Input: {os.path.basename(input_video_path)}")
        parts = []
        if 'width' in media_info:
            parts.append(f"{media_info['width']}x{media_info['height']}")
        if 'fps' in media_info:
            parts.append(f"{media_info['fps']} fps")
        if 'video_codec' in media_info:
            parts.append(media_info['video_codec'])
        if 'audio_codec' in media_info:
            parts.append(media_info['audio_codec'])
        if 'duration' in media_info:
            parts.append(format_duration(media_info['duration']))
        if 'size_bytes' in media_info:
            parts.append(format_file_size(media_info['size_bytes']))
        print(f"   {' | '.join(parts)}")
        total_frames_est = int(media_info.get('duration', 0) * media_info.get('fps', 0))
        if total_frames_est > 0:
            print(f"   ~{total_frames_est:,} frames to process")
    enc_label = f"{encoder_name} ({' '.join(enc_args)})"
    print(f"   Ratio: {ratio} | Quality: {quality} | Encoder: {enc_label}")
    print()

    # Pre-processing: normalize VFR to CFR if needed
    if is_variable_frame_rate(input_video):
        print("⚠️  Variable frame rate detected — normalizing to constant frame rate first...")
        duration = media_info.get('duration', 0) if media_info else 0
        if normalize_to_cfr(input_video, temp_cfr_input, total_duration=duration):
            input_video = temp_cfr_input
            print("✅ VFR normalization complete.")
        else:
            print("⚠️  Proceeding with original VFR file (audio sync may be affected).")

    print("🎬 Step 1: Detecting scenes...")
    step_start_time = time.time()
    scenes, _ = detect_scenes(input_video, downscale=downscale, frame_skip=frame_skip)
    step_end_time = time.time()
    
    if not scenes:
        print("⚠️  No scenes were detected. Treating entire video as one scene.")
        # Create fallback scene covering entire video duration
        from scenedetect import SceneManager
        media_info = get_media_info(input_video)
        duration = media_info.get('duration', 0)
        fps = media_info.get('fps', 30)
        
        if duration > 0:
            # Create a single scene covering the entire video
            class FakeScene:
                def __init__(self, start, end):
                    self.start = start
                    self.end = end
                def get_frames(self):
                    return int(self.start.get_seconds() * fps), int(self.end.get_seconds() * fps)
                def get_seconds(self):
                    return self.start.get_seconds(), self.end.get_seconds()
                def get_timecode(self):
                    return self.start.get_timecode(), self.end.get_timecode()
            
            start_time = SceneManager._create_scene(0, 0)
            end_time = SceneManager._create_scene(duration, duration)
            scenes = [FakeScene(start_time, end_time)]
        else:
            print("❌ Could not determine video duration. Aborting.")
            sys.exit(1)
    
    print(f"✅ Found {len(scenes)} scenes in {step_end_time - step_start_time:.2f}s. Here is the breakdown:")
    for i, (start, end) in enumerate(scenes):
        print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")


    print("\n🧠 Step 2: Analyzing scene content and determining strategy...")
    step_start_time = time.time()
    # Get fps from OpenCV — the same backend that reads the frames — to avoid
    # frame-rate mismatches between the reader and encoder that cause audio drift.
    original_width, original_height, fps = get_video_properties(input_video)
    
    OUTPUT_HEIGHT = original_height
    if OUTPUT_HEIGHT % 2 != 0:
        OUTPUT_HEIGHT += 1
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * ASPECT_RATIO)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    scenes_analysis = []
    for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing Scenes")):
        analysis = analyze_scene_content(input_video, start_time, end_time)
        strategy, target_box = decide_cropping_strategy(analysis, original_height)
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'start_seconds': start_time.get_seconds(),
            'end_seconds': end_time.get_seconds(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box
        })
    step_end_time = time.time()
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

    print("\n📋 Step 3: Generated Processing Plan")
    for i, scene_data in enumerate(scenes_analysis):
        num_people = len(scene_data['analysis'])
        strategy = scene_data['strategy']
        start_time = scenes[i][0].get_timecode()
        end_time = scenes[i][1].get_timecode()
        print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s). Strategy: {strategy}")

    if plan_only:
        track_count = sum(1 for s in scenes_analysis if s['strategy'] == 'TRACK')
        letterbox_count = sum(1 for s in scenes_analysis if s['strategy'] == 'LETTERBOX')
        elapsed = time.time() - script_start_time
        print(f"\n📊 Plan summary: {track_count} TRACK / {letterbox_count} LETTERBOX scenes")
        print(f"⏱️  Analysis took {elapsed:.1f}s. Run without plan_only=True to encode.")
        sys.exit(0)

    print("\n✂️ Step 4: Processing video frames...")
    step_start_time = time.time()

    command = [
        get_ffmpeg_path('ffmpeg'), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', encoder_name, *enc_args,
        '-pix_fmt', 'yuv420p',
        '-r', str(fps), '-vsync', 'cfr',
        '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    current_scene_index = 0
    dropped_frames = 0
    last_output_frame = None

    num_scenes = len(scenes_analysis)
    with tqdm(total=total_frames, desc=f"Processing [scene 1/{num_scenes}]",
              unit="fr", dynamic_ncols=True,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_scene_index < len(scenes_analysis) - 1 and \
               frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                current_scene_index += 1
                pbar.set_description(f"Processing [scene {current_scene_index + 1}/{num_scenes}]")

            scene_data = scenes_analysis[current_scene_index]
            strategy = scene_data['strategy']
            target_box = scene_data['target_box']

            try:
                # --- ALWAYS CREATE A BLURRED BACKGROUND FIRST ---
                # 1. Scale original frame to completely cover the output area
                bg_scale = max(OUTPUT_WIDTH / original_width, OUTPUT_HEIGHT / original_height)
                bg_w = int(original_width * bg_scale)
                bg_h = int(original_height * bg_scale)
                bg_frame = cv2.resize(frame, (bg_w, bg_h))
                
                # 2. Crop the center of the scaled background frame
                x_off = (bg_w - OUTPUT_WIDTH) // 2
                y_off = (bg_h - OUTPUT_HEIGHT) // 2
                bg_cropped = bg_frame[y_off:y_off + OUTPUT_HEIGHT, x_off:x_off + OUTPUT_WIDTH]
                
                # 3. Apply heavy blur and darken it
                output_frame = cv2.GaussianBlur(bg_cropped, (99, 99), 0)
                output_frame = cv2.convertScaleAbs(output_frame, alpha=0.5, beta=0)
                
                # --- NOW OVERLAY THE ACTUAL VIDEO ON TOP ---
                if strategy == 'TRACK':
                    # Get the cropped vertical slice of the video
                    crop_box = calculate_crop_box(target_box, original_width, original_height)
                    processed_frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    
                    # We only resize it to fit within the height, keeping its cropped aspect ratio
                    # (This prevents stretching if the math gets weird)
                    fg_h = OUTPUT_HEIGHT
                    fg_w = processed_frame.shape[1] 
                    
                    # If the cropped width is somehow wider than the output, scale it down
                    if fg_w > OUTPUT_WIDTH:
                        scale = OUTPUT_WIDTH / fg_w
                        fg_w = int(fg_w * scale)
                        fg_h = int(fg_h * scale)
                        
                    fg_frame = cv2.resize(processed_frame, (fg_w, fg_h))
                    
                    # Center the tracked frame over the blurred background
                    y_offset = (OUTPUT_HEIGHT - fg_h) // 2
                    x_offset = (OUTPUT_WIDTH - fg_w) // 2
                    output_frame[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = fg_frame

                else:  # LETTERBOX
                    # Scale the full horizontal video to fit the width
                    fg_scale = OUTPUT_WIDTH / original_width
                    fg_h = int(original_height * fg_scale)
                    fg_frame = cv2.resize(frame, (OUTPUT_WIDTH, fg_h))

                    # Center it vertically over the blurred background
                    y_offset = (OUTPUT_HEIGHT - fg_h) // 2
                    output_frame[y_offset:y_offset + fg_h, :] = fg_frame

                last_output_frame = output_frame
            except Exception:
                dropped_frames += 1
                if last_output_frame is not None:
                    output_frame = last_output_frame
                else:
                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)

    if dropped_frames > 0:
        print(f"  ⚠️  {dropped_frames} frame(s) could not be processed and were duplicated from the previous frame.")

    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n❌ FFmpeg frame processing failed.")
        print("Stderr:", stderr_output)
        cleanup_temp_files()
        sys.exit(1)
    step_end_time = time.time()
    print(f"✅ Video processing complete in {step_end_time - step_start_time:.2f}s.")

    input_has_audio = has_audio_stream(input_video)

    if input_has_audio:
        print("\n🔊 Step 5: Extracting original audio...")
        step_start_time = time.time()

        # Some files have a non-zero video start_time (e.g. audio starts at 0s
        # but video starts at 1.8s). OpenCV ignores this offset and reads frames
        # from the first video frame, so the processed video starts at 0s.
        # We must trim the audio to match: skip audio before the video started,
        # and limit to the video's duration.
        video_start = get_stream_start_time(input_video, 'v:0')
        audio_extract_command = [
            get_ffmpeg_path('ffmpeg'), '-y', '-ss', str(video_start),
            '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
        ]
        try:
            subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Audio extracted in {step_end_time - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Audio extraction failed.")
            print("Stderr:", e.stderr.decode())
            cleanup_temp_files()
            sys.exit(1)

        print("\n✨ Step 6: Merging video and audio...")
        step_start_time = time.time()
        merge_command = [
            get_ffmpeg_path('ffmpeg'), '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', '-shortest', final_output_video
        ]
        try:
            subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Final video merged in {step_end_time - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Final merge failed.")
            print("Stderr:", e.stderr.decode())
            cleanup_temp_files()
            sys.exit(1)

        cleanup_temp_files()
    else:
        print("\n🔇 Step 5: No audio stream detected, skipping audio extraction.")
        # Just rename the temp video as the final output
        os.rename(temp_video_output, final_output_video)
        cleanup_temp_files()

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    # Final summary
    print(f"\n{'─' * 50}")
    print(f"🎉 All done! Final video saved to {final_output_video}")
    print(f"{'─' * 50}")
    output_info = get_media_info(final_output_video)
    if output_info:
        out_parts = []
        if 'width' in output_info:
            out_parts.append(f"{output_info['width']}x{output_info['height']}")
        if 'duration' in output_info:
            out_parts.append(format_duration(output_info['duration']))
        if 'size_bytes' in output_info:
            out_parts.append(format_file_size(output_info['size_bytes']))
        print(f"   Output: {' | '.join(out_parts)}")
    if media_info and output_info and media_info.get('size_bytes') and output_info.get('size_bytes'):
        ratio = output_info['size_bytes'] / media_info['size_bytes'] * 100
        print(f"   Size:   {format_file_size(media_info['size_bytes'])} → {format_file_size(output_info['size_bytes'])} ({ratio:.0f}% of original)")
    print(f"   Time:   {format_duration(total_time)} ({total_time:.1f}s)")
    if media_info and media_info.get('duration'):
        speed = media_info['duration'] / total_time if total_time > 0 else 0
        print(f"   Speed:  {speed:.1f}x real-time")

if __name__ == '__main__':
    # Ensure LOCAL_BIN_DIR is available
    original_path = os.environ.get("PATH", "")
    if LOCAL_BIN_DIR not in original_path:
        os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
        print(f"🔧 Added {LOCAL_BIN_DIR} to PATH for CropperM module")
    
    parser = argparse.ArgumentParser(description="Smartly crops a horizontal video into a vertical one.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output video file.")
    parser.add_argument('--ratio', type=str, default='9:16',
                        help="Output aspect ratio as W:H (default: 9:16). Examples: 9:16, 4:5, 1:1")
    parser.add_argument('--quality', type=str, default='balanced', choices=['fast', 'balanced', 'high'],
                        help="Encoding quality preset (default: balanced). fast=quick encode, balanced=good quality, high=best quality/slow")
    parser.add_argument('--crf', type=int, default=None,
                        help="Override CRF value directly (0-51, lower=better quality). Overrides --quality.")
    parser.add_argument('--preset', type=str, default=None,
                        help="Override FFmpeg x264 preset directly (ultrafast..veryslow). Overrides --quality.")
    parser.add_argument('--plan-only', action='store_true',
                        help="Only run scene detection and analysis (Steps 1-3), then print the processing plan without encoding.")
    parser.add_argument('--frame-skip', type=int, default=0,
                        help="Frames to skip during scene detection (default: 0 = every frame, most accurate). "
                             "1 = every other frame (~2x faster). Higher = faster but may miss quick cuts.")
    parser.add_argument('--downscale', type=int, default=0,
                        help="Downscale factor for scene detection (default: 0 = auto). "
                             "Higher values (2-4) are faster but may miss subtle scene changes.")
    parser.add_argument('--encoder', type=str, default='auto',
                        help="Video encoder: 'auto' (libx264, default), 'hw' (auto-detect hardware encoder), "
                             "or a specific encoder name like 'h264_videotoolbox' or 'h264_nvenc'.")
    args = parser.parse_args()
    
    main(
        input_video_path=args.input,
        output_video_path=args.output,
        ratio=args.ratio,
        quality=args.quality,
        crf=args.crf,
        preset=args.preset,
        plan_only=args.plan_only,
        frame_skip=args.frame_skip,
        downscale=args.downscale,
        encoder=args.encoder
    )
