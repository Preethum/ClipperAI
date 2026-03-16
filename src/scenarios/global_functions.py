"""
Global functions — Pipeline orchestrator using Plan-Then-Render architecture.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types from YOLO/scenedetect in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

# Set working directory to project root
def set_project_root_as_cwd():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)
    return project_root

PROJECT_ROOT = set_project_root_as_cwd()

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_bin_dir, setup_bin_path, get_templates_dir, get_output_dir
except ImportError:
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def get_bin_dir():
        return os.path.join(get_project_root(), 'bin')
    def get_templates_dir():
        return os.path.join(get_project_root(), 'templates')
    def get_output_dir():
        return os.path.join(get_project_root(), 'output')
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

def _ensure_bin_path():
    p = os.environ.get("PATH", "")
    if LOCAL_BIN_DIR not in p:
        os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + p

# Add module directory to path and import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from ClipperM import main as clipper_main
from CropperM import plan as cropper_plan
from SubsM import plan as subs_plan
from RendererM import main as renderer_main


def _header(step, icon, title, total_steps=4):
    """Print a clean step header."""
    bar = "━" * 50
    print(f"\n{bar}")
    print(f"  {icon}  Step {step}/{total_steps}: {title}")
    print(bar)


def _status(msg):
    """Print an indented status line."""
    print(f"     {msg}")


def run_complete_pipeline(config):
    """Run the Plan-Then-Render pipeline."""
    source_video = config.get("input_video")
    output_dir = config.get("base_output_dir", "output")
    modules = config.get("modules", {})

    os.makedirs(output_dir, exist_ok=True)
    _ensure_bin_path()

    # Count active steps
    steps = []
    clip_cfg = modules.get("clipper", {})
    crop_cfg = modules.get("cropper", {})
    subs_cfg = modules.get("subs", {})
    renderer_cfg = modules.get("renderer", {})

    if clip_cfg.get("enabled", True):   steps.append("clipper")
    if crop_cfg.get("enabled", True):   steps.append("cropper")
    if subs_cfg.get("enabled", True):   steps.append("subs")
    if renderer_cfg.get("enabled", True): steps.append("renderer")

    total = len(steps)
    step_num = 0
    pipeline_start = time.time()

    print(f"\n{'━' * 50}")
    print(f"  🚀  ClipperAI Pipeline")
    print(f"{'━' * 50}")
    _status(f"Source:  {os.path.basename(source_video)}")
    _status(f"Output:  {output_dir}")
    _status(f"Steps:   {' → '.join(s.upper() for s in steps)}")

    try:
        manifest = []
        vision_frame_timestamps = []

        # ── CLIPPER (produces manifest) ──
        if "clipper" in steps:
            step_num += 1
            _header(step_num, "🎬", "CLIPPER — Finding clips", total)
            _status(f"Scout: {clip_cfg.get('scout_model', '?')}  |  Editor: {clip_cfg.get('editor_model', '?')}")
            t0 = time.time()
            manifest, vision_frame_timestamps = clipper_main(
                input_video_path=source_video,
                output_dir=output_dir,
                **{k: clip_cfg[k] for k in (
                    'lm_studio_url', 'scout_model', 'editor_model',
                    'min_clip_duration', 'max_clip_duration', 'max_total_clips',
                    'viral_archetypes', 'scout_system_instruction', 'scout_user_prompt',
                    'editor_system_instruction', 'editor_user_prompt',
                    'deduplication_threshold', 'enable_ocr', 'enable_vision',
                    'vision_model', 'vision_interval', 'vision_concurrency'
                )}
            )
            _status(f"✅ {len(manifest)} clips found ({time.time()-t0:.0f}s)")

        if not manifest:
            print("\n  ❌ No clips found. Stopping.")
            return []

        # ── CROPPER (enrich manifest) ──
        if "cropper" in steps:
            step_num += 1
            _header(step_num, "✂️", f"CROPPER — Analyzing scenes ({crop_cfg.get('ratio', '9:16')})", total)
            t0 = time.time()
            # Pass vision frame timestamps to CropperM for multi-frame YOLO
            crop_cfg_with_frames = dict(crop_cfg)
            if vision_frame_timestamps:
                crop_cfg_with_frames["vision_frame_timestamps"] = vision_frame_timestamps
            manifest = cropper_plan(source_video, manifest, crop_cfg_with_frames)
            _status(f"✅ {len(manifest)} clips enriched ({time.time()-t0:.0f}s)")

        # ── SUBS (enrich manifest) ──
        if "subs" in steps:
            step_num += 1
            _header(step_num, "📝", "SUBS — Transcribing clips", total)
            t0 = time.time()
            manifest = subs_plan(source_video, manifest, subs_cfg)
            _status(f"✅ {len(manifest)} clips transcribed ({time.time()-t0:.0f}s)")

        # Save manifest checkpoint
        manifest_path = os.path.join(output_dir, "clips_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4, cls=_NumpyEncoder)

        # ── RENDERER (execute) ──
        if "renderer" in steps:
            step_num += 1
            _header(step_num, "🎬", "RENDERER — Exporting final clips", total)
            t0 = time.time()
            final_clips = renderer_main(source_video, manifest, output_dir, renderer_cfg)
            _status(f"✅ {len(final_clips)} clips rendered ({time.time()-t0:.0f}s)")
        else:
            final_clips = []

        # ── DONE ──
        elapsed = time.time() - pipeline_start
        print(f"\n{'━' * 50}")
        print(f"  🎉  Pipeline Complete — {len(final_clips)} clips in {elapsed:.0f}s")
        print(f"{'━' * 50}")
        for clip in final_clips:
            _status(f"📹 {clip.get('file_name', '?')}  ({clip.get('duration', 0):.0f}s)  {clip.get('title', '')}")
        _status(f"📁 {output_dir}")

        return final_clips

    except Exception as e:
        print(f"\n  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_gaming_pipeline(config):
    """Run the complete YOLO/OCR gaming pipeline using global configuration."""
    source_video = config.get("input_video")
    output_dir = config.get("base_output_dir", "output")
    modules = config.get("modules", {})
    
    os.makedirs(output_dir, exist_ok=True)
    _ensure_bin_path()
    
    detector_cfg = modules.get("detector", {})
    audio_cfg = modules.get("audio", {})
    clumper_cfg = modules.get("clumper", {})
    safe_entry_cfg = modules.get("safe_entry_exit", {})
    renderer_cfg = modules.get("renderer", {})
    
    steps = []
    if detector_cfg.get("enabled", True): steps.append("detection")
    if audio_cfg.get("enabled", True): steps.append("audio")
    if clumper_cfg.get("enabled", True): steps.append("clumping")
    if safe_entry_cfg.get("enabled", True): steps.append("pacing")
    if renderer_cfg.get("enabled", True): steps.append("renderer")
    
    total = len(steps)
    step_num = 0
    pipeline_start = time.time()
    
    print(f"\n{'━' * 50}")
    print(f"  🎮  Gaming Highlight Pipeline")
    print(f"{'━' * 50}")
    _status(f"Source:  {os.path.basename(source_video)}")
    _status(f"Output:  {output_dir}")
    _status(f"Steps:   {' → '.join(s.upper() for s in steps)}")

    try:
        events_json = os.path.join(output_dir, "events.json")
        audio_events_json = os.path.join(output_dir, "audio_events.json")
        clips_json = os.path.join(output_dir, "clips.json")
        
        audio_analyzer = None
        audio_events = []

        # ── AUDIO ANALYSIS (Moved to Step 1 for pruning) ──
        if "audio" in steps:
            step_num += 1
            _header(step_num, "🔊", "AUDIO — Combat Intensity Scoring", total)
            t0 = time.time()
            try:
                from AudioAnalyzerM import AudioAnalyzer
                audio_analyzer = AudioAnalyzer(
                    video_path=source_video,
                    chunk_sec=audio_cfg.get("chunk_sec", 1.0),
                    device=audio_cfg.get("device", 0),
                    score_threshold=audio_cfg.get("threshold", 0.3)
                )
                
                audio_events = audio_analyzer.process_gameplay(output_path=audio_events_json)
                _status(f"✅ Audio processed: {len(audio_events)} combat spikes found ({time.time()-t0:.0f}s)")
            except Exception as e:
                _status(f"⚠️ Audio analysis failed: {e}. Falling back to full visual scan.")

        # ── DETECTION (vision + OCR) ──
        if "detection" in steps:
            step_num += 1
            _header(step_num, "🎯", "DETECTION — YOLO & OCR Scanning", total)
            t0 = time.time()
            from DetectionM import GameEventDetector
            detector = GameEventDetector(detector_cfg.get("model_path"), enable_ocr=detector_cfg.get("enable_ocr", True))
            _status(f"YOLO Model: {detector_cfg.get('model_path')}")
            
            # Create sampling windows from audio if available
            scan_windows = None
            if audio_events:
                scan_windows = _get_scan_windows(audio_events, padding=15.0)
                _status(f"Audio-Guided Pruning: Scanning only {len(scan_windows)} active segments.")

            all_events = detector.detect_events(
                source_video, 
                interval_seconds=detector_cfg.get("interval_seconds", 1.0),
                min_confidence=detector_cfg.get("min_confidence", 0.5),
                target_labels=detector_cfg.get("target_labels", ["Victory", "ELIMINATED Text", "loating Damage Numbers", "Directional Damage Indicators"]),
                label_weights=detector_cfg.get("label_weights", None),
                debug=detector_cfg.get("debug", False),
                include_windows=scan_windows
            )
            
            # Merge audio events into the result set
            if audio_events and audio_cfg.get("merge_to_events", True):
                for ts, label, score in audio_events:
                    all_events.append({"timestamp_s": ts, "label": label, "confidence": score, "text": ""})
            
            all_events.sort(key=lambda e: e.get('timestamp_s', 0))
            
            with open(events_json, 'w', encoding='utf-8') as f:
                json.dump(all_events, f, indent=2, ensure_ascii=False)
            _status(f"✅ {len(all_events)} total events found ({time.time()-t0:.0f}s)")
            
        if not os.path.exists(events_json):
            print("\n  ❌ No valid events.json found. Stopping.")
            return []

        # ── CLUMPING ──
        if "clumping" in steps:
            step_num += 1
            _header(step_num, "🎬", "CLUMPING — Building Highlights", total)
            t0 = time.time()
            from Clipper_gamingM import EventClumper
            clumper = EventClumper(events_json, clumper_cfg.get("window_seconds", 60), clumper_cfg.get("intensity_weights", None))
            
            clumper_mode = clumper_cfg.get("mode", "sliding_window")
            if clumper_mode == "elimination" and audio_events:
                _status("Mode: Audio-Anchored Elimination Clips")
                clips = clumper.detect_elimination_clips(
                    audio_events=audio_events,
                    elim_merge_gap=clumper_cfg.get("elim_merge_gap", 30.0),
                    audio_lookback=clumper_cfg.get("audio_lookback", 30.0),
                    audio_lookahead=clumper_cfg.get("audio_lookahead", 30.0),
                    end_buffer=clumper_cfg.get("end_buffer", 5.0),
                    clip_merge_gap=clumper_cfg.get("clip_merge_gap", 60.0),
                )
            else:
                clips = clumper.detect_clips(clumper_cfg.get("min_intensity", 3.0), clumper_cfg.get("gap_threshold", 30.0))
            
            clumper.save_clips(clips_json)
            _status(f"✅ {len(clips)} highlights assembled ({time.time()-t0:.0f}s)")
            
        # ── PACING (SAFE ENTRY/EXIT) ──
        if "pacing" in steps:
            step_num += 1
            _header(step_num, "🔍", "PACING — Fine-tuning Cut Points", total)
            t0 = time.time()
            from SafeEntryExit import SafeEntryExitDetector
            pacing_agent = SafeEntryExitDetector(
                video_path=source_video,
                vlm_model=safe_entry_cfg.get("vlm_model", "google/gemma-3-27b"),
                fps=safe_entry_cfg.get("vlm_fps", 10),
                audio_analyzer=audio_analyzer
            )
            try:
                clips = pacing_agent.process_clips(clips_json, safe_entry_cfg.get("entry_buffer", 1.2), safe_entry_cfg.get("exit_buffer", 1.5))
                _status(f"✅ {len(clips)} clips safely padded ({time.time()-t0:.0f}s)")
            finally:
                pacing_agent.close()
                
            # Plot overlay if audio exists
            if audio_analyzer:
                try:
                    with open(events_json, 'r', encoding='utf-8') as f:
                        ev_data = json.load(f)
                    audio_len_sec = len(audio_analyzer._audio) / audio_analyzer._sr
                    rms_profile = audio_analyzer.get_audio_intensity_profile(0, audio_len_sec, resolution=1.0)
                    plot_path = os.path.join(output_dir, "audio_events_intensity.png")
                    audio_analyzer.save_intensity_plot(rms_profile, ev_data, plot_path, clips)
                except Exception as e:
                    pass

        # ── RENDERER ──
        final_clips = []
        if "renderer" in steps:
            step_num += 1
            _header(step_num, "🎥", "RENDERER — Exporting Final Videos", total)
            t0 = time.time()
            
            # Read clips.json and convert to RendererM native manifest format 
            with open(clips_json, 'r', encoding='utf-8') as f:
                raw_clips = json.load(f)
                
            manifest = []
            for i, c in enumerate(raw_clips):
                st = c.get('safe_entry', {}).get('timestamp') or c['start_time']
                et = c.get('safe_exit', {}).get('timestamp') or c['end_time']
                manifest.append({
                    "start": st, "end": et, "file_name": f"clip_{i+1}.mp4",
                    "title": c.get('clip_id', f"Clip {i+1}"), "scores": c.get('scores', {})
                })
            
            # Execute RendererM module natively
            clips_output_dir = os.path.join(output_dir, "video_clips")
            os.makedirs(clips_output_dir, exist_ok=True)
            final_clips = renderer_main(source_video, manifest, clips_output_dir, renderer_cfg)
            _status(f"✅ {len(final_clips)} videos rendered ({time.time()-t0:.0f}s)")

        # ── DONE ──
        elapsed = time.time() - pipeline_start
        print(f"\n{'━' * 50}")
        print(f"  🎉  Gaming Pipeline Complete — {len(final_clips) if final_clips else len(clips)} clips in {elapsed:.0f}s")
        print(f"{'━' * 50}")
        _status(f"📁 {output_dir}")
        return final_clips

    except Exception as e:
        print(f"\n  ❌ Gaming Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def _get_scan_windows(events: List, padding: float = 15.0) -> List[Tuple[float, float]]:
    """Convert audio combat spikes into merged scanning windows for visual analysis."""
    if not events:
        return []
    
    # Extract timestamps from list of [ts, label, score]
    timestamps = [e[0] for e in events]
    raw_windows = [(ts - padding, ts + padding) for ts in timestamps]
    raw_windows.sort()
    
    if not raw_windows:
        return []
    
    merged = []
    curr_start, curr_end = raw_windows[0]
    
    for next_start, next_end in raw_windows[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((max(0, curr_start), curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged.append((max(0, curr_start), curr_end))
    return merged
