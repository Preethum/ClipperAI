import os
import cv2
import re
import json
import shutil
import pycaps
from pycaps import TemplateLoader, VideoQuality

# Import path utilities
try:
    from utils.path_utils import get_project_root, get_templates_dir
except ImportError:
    # Fallback if utils module not available
    def get_project_root():
        # More robust fallback - try multiple methods
        current_file = os.path.abspath(__file__)
        # We're in src/modules, so go up 2 levels to project root
        return os.path.dirname(os.path.dirname(current_file))
    def get_templates_dir():
        return os.path.join(get_project_root(), 'templates')

# --- CONFIGURATION ---
# Get project root for relative paths
PROJECT_ROOT = get_project_root()
DEFAULT_VIDEO_INPUT = os.path.join(PROJECT_ROOT, "test.mp4")
DEFAULT_VIDEO_OUTPUT = os.path.join(PROJECT_ROOT, "output", "veout.mp4")
DEFAULT_ORIGINAL_TEMPLATE = os.path.join(get_templates_dir(), "default")
DEFAULT_TEMP_TEMPLATE_PATH = os.path.join(get_templates_dir(), "temp_smart_merge")

# Runtime configuration (will be set by main function)
video_input = DEFAULT_VIDEO_INPUT
video_output = DEFAULT_VIDEO_OUTPUT
original_template = DEFAULT_ORIGINAL_TEMPLATE
temp_template_path = DEFAULT_TEMP_TEMPLATE_PATH

def main(input_video_path=None, output_video_path=None, template_path=None, 
         vertical_align_offset=0.70, max_width_ratio=0.9, max_lines=1):
    """
    Main function to run subtitle rendering with custom configuration.
    
    Args:
        input_video_path (str): Path to input video file
        output_video_path (str): Path to output video file
        template_path (str): Path to template directory
        vertical_align_offset (float): Vertical alignment offset (0.0-1.0)
        max_width_ratio (float): Maximum width ratio for subtitles
        max_lines (int): Maximum number of lines for subtitles
    
    Returns:
        bool: True if successful, False otherwise
    """
    global video_input, video_output, original_template, temp_template_path
    
    # Update configuration
    video_input = input_video_path or DEFAULT_VIDEO_INPUT
    video_output = output_video_path or DEFAULT_VIDEO_OUTPUT
    original_template = template_path or DEFAULT_ORIGINAL_TEMPLATE
    temp_template_path = os.path.join(original_template, "temp_" + str(hash(video_input) % 10000))
    
    return run_render()

def run_render():
    """Main subtitle rendering function."""
    global video_input, video_output, original_template, temp_template_path

    # 1. Get video dimensions
    if not os.path.exists(video_input):
        print(f"❌ Video not found: {video_input}")
        return
    
    cap = cv2.VideoCapture(video_input)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Target: Bottom 1/3 (66% to 87.5% height)
    usable_box_height = h * (0.875 - 0.66)
    dynamic_font_size = int(usable_box_height / 1.6) 
    dynamic_font_size = max(10, dynamic_font_size)  # Absolute minimum

    # 2. Setup Temporary Workspace
    if os.path.exists(temp_template_path):
        shutil.rmtree(temp_template_path)
    shutil.copytree(original_template, temp_template_path)
    
    # 4. Configure JSON layout
    json_path = os.path.join(temp_template_path, "pycaps.template.json")
    css_path = os.path.join(temp_template_path, "style.css")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove output key to prevent JSON export
    if "output" in data:
        del data["output"]
    
    if "layout" not in data:
        data["layout"] = {}
    
    # Update Layout: Top alignment with 0.70 offset
    data["layout"].update({
        "vertical_align": {
            "align": "top",
            "offset": 0.70
        },
        "max_width_ratio": 0.9,
        "max_number_of_lines": 1
    })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    # 5. Update CSS with PyCaps compatibility
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Update font size
        css_content = re.sub(r"font-size\s*:\s*\d+px", f"font-size: {dynamic_font_size}px", css_content)
        
        # Add minimum width to prevent PyCaps clipping
        if ".word" not in css_content:
            css_content += "\n\n/* PyCaps compatibility fixes */\n"
            css_content += ".word {\n"
            css_content += f"    min-width: {max(25, dynamic_font_size)}px !important;\n"
            css_content += "    display: inline-block;\n"
            css_content += "    white-space: nowrap;\n"
            css_content += "}\n"
        
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)

    print(f"📐 Video: {w}x{h} | Font: {dynamic_font_size}px")

    # 6. Render video
    render_success = False
    try:
        print(f"🔍 Loading template from: {temp_template_path}")
        loader = TemplateLoader(temp_template_path)
        builder = loader.load(False)
        
        print(f"📹 Input video: {video_input}")
        print(f"📹 Output video: {video_output}")
        
        builder.with_input_video(video_input)
        builder.with_output_video(video_output)
        builder.with_video_quality(VideoQuality.HIGH)
        
        print(f"🚀 Rendering...")
        builder.build().run()
        
        if os.path.exists(video_output):
            print(f"✅ Render Successful: {video_output}")
            render_success = True
            
            # Clean up any sidecar JSON
            potential_json = video_output.replace(".mp4", ".json")
            if os.path.exists(potential_json):
                os.remove(potential_json)
                print(f"🧹 Removed sidecar JSON: {potential_json}")
        else:
            print(f"❌ Output file not created: {video_output}")
                
    except Exception as e:
        print(f"❌ Render failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        if os.path.exists(temp_template_path):
            shutil.rmtree(temp_template_path)
    
    return render_success

if __name__ == "__main__":
    run_render()