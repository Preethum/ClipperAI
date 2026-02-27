"""
Global functions — Pipeline orchestrator using Plan-Then-Render architecture.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path


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

        # ── CLIPPER (produces manifest) ──
        if "clipper" in steps:
            step_num += 1
            _header(step_num, "🎬", "CLIPPER — Finding clips", total)
            _status(f"Scout: {clip_cfg.get('scout_model', '?')}  |  Editor: {clip_cfg.get('editor_model', '?')}")
            t0 = time.time()
            manifest = clipper_main(
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
            manifest = cropper_plan(source_video, manifest, crop_cfg)
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
