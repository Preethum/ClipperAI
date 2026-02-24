import os
import cv2
import re
import json
import shutil
import pycaps
from pycaps import TemplateLoader, VideoQuality
from typing import Optional, Tuple

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_templates_dir
except ImportError:
    def get_project_root():
        current_file = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(current_file))
    def get_templates_dir():
        return os.path.join(get_project_root(), 'templates')

# --- CONFIGURATION ---
PROJECT_ROOT = get_project_root()
DEFAULT_VIDEO_INPUT = os.path.join(PROJECT_ROOT, "test.mp4")
DEFAULT_VIDEO_OUTPUT = os.path.join(PROJECT_ROOT, "output", "veout.mp4")
DEFAULT_ORIGINAL_TEMPLATE = os.path.join(get_templates_dir(), "default")
DEFAULT_TEMP_TEMPLATE_PATH = os.path.join(get_templates_dir(), "temp_smart_merge")

def _get_video_dimensions(video_path: str) -> Optional[Tuple[int, int]]:
    """Get video dimensions (width, height)."""
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    try:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return w, h
    finally:
        cap.release()

def _calculate_font_size(height: int) -> int:
    """Calculate dynamic font size based on video height."""
    usable_box_height = height * (0.875 - 0.66)
    dynamic_font_size = int(usable_box_height / 1.6)
    return max(10, dynamic_font_size)

def _setup_template_workspace(original_template: str, temp_template_path: str) -> None:
    """Setup temporary template workspace."""
    if os.path.exists(temp_template_path):
        shutil.rmtree(temp_template_path)
    shutil.copytree(original_template, temp_template_path)

def _configure_template_json(json_path: str, vertical_align_offset: float, max_width_ratio: float, max_lines: int) -> None:
    """Configure template JSON with layout settings."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove output key to prevent JSON export
    data.pop("output", None)
    
    # Ensure layout exists
    if "layout" not in data:
        data["layout"] = {}
    
    # Update layout configuration
    data["layout"].update({
        "vertical_align": {
            "align": "top",
            "offset": vertical_align_offset
        },
        "max_width_ratio": max_width_ratio,
        "max_number_of_lines": max_lines
    })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def _update_template_css(css_path: str, font_size: int) -> None:
    """Update template CSS with dynamic font size and PyCaps compatibility."""
    if not os.path.exists(css_path):
        return
        
    with open(css_path, 'r', encoding='utf-8') as f:
        css_content = f.read()
    
    # Update font size
    css_content = re.sub(r"font-size\s*:\s*\d+px", f"font-size: {font_size}px", css_content)
    
    # Add PyCaps compatibility fixes if not present
    if ".word" not in css_content:
        css_content += f"\n\n/* PyCaps compatibility fixes */\n"
        css_content += ".word {\n"
        css_content += f"    min-width: {max(25, font_size)}px !important;\n"
        css_content += "    display: inline-block;\n"
        css_content += "    white-space: nowrap;\n"
        css_content += "}\n"
    
    with open(css_path, 'w', encoding='utf-8') as f:
        f.write(css_content)

def _render_video(template_path: str, input_video: str, output_video: str) -> bool:
    """Render video using PyCaps template."""
    try:
        print(f"🔍 Loading template from: {template_path}")
        loader = TemplateLoader(template_path)
        builder = loader.load(False)
        
        print(f"📹 Input video: {input_video}")
        print(f"📹 Output video: {output_video}")
        
        builder.with_input_video(input_video)
        builder.with_output_video(output_video)
        builder.with_video_quality(VideoQuality.HIGH)
        
        print(f"🚀 Rendering...")
        builder.build().run()
        
        if os.path.exists(output_video):
            print(f"✅ Render Successful: {output_video}")
            
            # Clean up sidecar JSON
            potential_json = output_video.replace(".mp4", ".json")
            if os.path.exists(potential_json):
                os.remove(potential_json)
                print(f"🧹 Removed sidecar JSON: {potential_json}")
            return True
        else:
            print(f"❌ Output file not created: {output_video}")
            return False
            
    except Exception as e:
        print(f"❌ Render failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def main(input_video_path: Optional[str] = None, output_video_path: Optional[str] = None, template_path: Optional[str] = None, 
         vertical_align_offset: float = 0.70, max_width_ratio: float = 0.9, max_lines: int = 1) -> bool:
    """Main function to run subtitle rendering with custom configuration."""
    # Setup configuration
    video_input = input_video_path or DEFAULT_VIDEO_INPUT
    video_output = output_video_path or DEFAULT_VIDEO_OUTPUT
    original_template = template_path or DEFAULT_ORIGINAL_TEMPLATE
    temp_template_path = os.path.join(original_template, f"temp_{hash(video_input) % 10000}")
    
    # Get video dimensions
    dimensions = _get_video_dimensions(video_input)
    if not dimensions:
        return False
    
    w, h = dimensions
    font_size = _calculate_font_size(h)
    
    print(f"📐 Video: {w}x{h} | Font: {font_size}px")
    
    # Setup template workspace
    _setup_template_workspace(original_template, temp_template_path)
    
    # Configure template files
    json_path = os.path.join(temp_template_path, "pycaps.template.json")
    css_path = os.path.join(temp_template_path, "style.css")
    
    _configure_template_json(json_path, vertical_align_offset, max_width_ratio, max_lines)
    _update_template_css(css_path, font_size)
    
    # Render video
    success = _render_video(temp_template_path, video_input, video_output)
    
    # Cleanup
    if os.path.exists(temp_template_path):
        shutil.rmtree(temp_template_path)
    
    return success

def run_render() -> bool:
    """Legacy function for backward compatibility."""
    return main()

if __name__ == "__main__":
    run_render()