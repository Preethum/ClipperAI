import cv2
import easyocr
import os
import json
import json_repair
import re
import difflib
import requests
import numpy as np
from openai import OpenAI
from faster_whisper import WhisperModel
from tqdm import tqdm 
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import path utilities
try:
    import sys
    import os
    # Add src to path so we can import utils
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from utils.path_utils import get_project_root, get_bin_dir, setup_bin_path, get_output_dir
except ImportError as e:
    print(f"ImportError: {e}")
    # Fallback if utils module not available
    def get_project_root():
        # Direct fallback - go up 2 levels from modules to project root
        current_file = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(current_file))
    def get_bin_dir():
        project_root = get_project_root()
        bin_path = os.path.join(project_root, 'bin')
        print(f"Fallback bin path: {bin_path}")
        return bin_path
    def get_output_dir():
        return os.path.join(get_project_root(), 'output')
    def setup_bin_path():
        bin_dir = get_bin_dir()
        if os.path.exists(bin_dir):
            current_path = os.environ.get("PATH", "")
            if bin_dir not in current_path:
                os.environ["PATH"] = bin_dir + os.pathsep + current_path
            print(f"✅ Added to PATH: {bin_dir}")
            return bin_dir
        else:
            print(f"❌ Warning: Bin directory not found at {bin_dir}")
            return None

# Add folder to PATH for external libraries (like FFmpeg)
LOCAL_BIN_DIR = setup_bin_path()
if LOCAL_BIN_DIR is None:
    LOCAL_BIN_DIR = get_bin_dir()  # For reference in error messages

# --- DEFAULT CONFIGURATION ---
# Get project root: src/modules -> src -> project root
PROJECT_ROOT = get_project_root()
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "podcast.mp4")
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1" 

# Multi-Agent Pipeline models
DEFAULT_SCOUT_MODEL = "deepseek-r1-distill-qwen-32b" # Pass 1: Logical filtering & data crunching
DEFAULT_EDITOR_MODEL = "google/gemma-3-27b"          # Pass 2: Creative narrative selection

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out", "01_clips")

# --- SYSTEM ADJUSTMENT VARIABLES ---
DEFAULT_MIN_CLIP_DURATION = 45.0  # Minimum length of a clip in seconds
DEFAULT_MAX_CLIP_DURATION = 90.0  # Maximum length of a clip in seconds
DEFAULT_MAX_TOTAL_CLIPS = 40       # Maximum number of total clips you want the pipeline to output

# Runtime configuration (will be set by main function)
VIDEO_PATH = DEFAULT_VIDEO_PATH
LM_STUDIO_URL = DEFAULT_LM_STUDIO_URL
SCOUT_MODEL = DEFAULT_SCOUT_MODEL
EDITOR_MODEL = DEFAULT_EDITOR_MODEL
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MIN_CLIP_DURATION = DEFAULT_MIN_CLIP_DURATION
MAX_CLIP_DURATION = DEFAULT_MAX_CLIP_DURATION
MAX_TOTAL_CLIPS = DEFAULT_MAX_TOTAL_CLIPS

client = None

# Initialize client with default values


def initialize_client():
    """Initialize the OpenAI client with current LM_STUDIO_URL."""
    global client
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=None)
initialize_client()
def main(input_video_path=None, output_dir=None, lm_studio_url=None, scout_model=None, 
         editor_model=None, min_clip_duration=None, max_clip_duration=None, max_total_clips=None):
    """
    Main function to run the clipper with custom configuration.
    
    Args:
        input_video_path (str): Path to input video file
        output_dir (str): Directory to save output clips
        lm_studio_url (str): LM Studio API URL
        scout_model (str): Model name for scout pass
        editor_model (str): Model name for editor pass
        min_clip_duration (float): Minimum clip duration in seconds
        max_clip_duration (float): Maximum clip duration in seconds
        max_total_clips (int): Maximum number of clips to generate
    
    Returns:
        list: List of generated clip metadata
    """
    global VIDEO_PATH, LM_STUDIO_URL, SCOUT_MODEL, EDITOR_MODEL, OUTPUT_DIR
    global MIN_CLIP_DURATION, MAX_CLIP_DURATION, MAX_TOTAL_CLIPS, client
    
    # Update configuration
    VIDEO_PATH = input_video_path or DEFAULT_VIDEO_PATH
    OUTPUT_DIR = output_dir or DEFAULT_OUTPUT_DIR
    LM_STUDIO_URL = lm_studio_url or DEFAULT_LM_STUDIO_URL
    SCOUT_MODEL = scout_model or DEFAULT_SCOUT_MODEL
    EDITOR_MODEL = editor_model or DEFAULT_EDITOR_MODEL
    MIN_CLIP_DURATION = min_clip_duration or DEFAULT_MIN_CLIP_DURATION
    MAX_CLIP_DURATION = max_clip_duration or DEFAULT_MAX_CLIP_DURATION
    MAX_TOTAL_CLIPS = max_total_clips or DEFAULT_MAX_TOTAL_CLIPS
    
    # Initialize client with new URL
    initialize_client()
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    # Run the clipper logic
    return _run_clipper_logic()

def _run_clipper_logic():
    """Execute the core clipper logic with current global configuration."""
    import cv2
    import easyocr
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from faster_whisper import WhisperModel
    
    # Initialize models
    reader = easyocr.Reader(['en'], gpu=True)
    whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
    
    # Get video duration
    with VideoFileClip(VIDEO_PATH) as video: 
        total_duration = video.duration
    
    # 1. Extraction
    print(" Extracting transcript and OCR data...")
    full_transcript = get_transcript_with_signals(VIDEO_PATH, whisper_model)
    full_ocr = get_ocr_list(VIDEO_PATH, reader)
    
    # 2. Generate semantic chapters
    print(" Analyzing semantic chapters...")
    semantic_chapters = generate_semantic_chapters(full_transcript, full_ocr)
    print(f"Found {len(semantic_chapters)} logical chapters.")
    
    # 3. Pass 1: Semantic Scouting
    raw_candidate_pool = []
    
    for i, (start_sec, end_sec) in enumerate(semantic_chapters):
        window_label = f"Chapter {i+1} ({format_time(start_sec)} to {format_time(end_sec)})"
        
        # Get context windows
        prev_start = max(0, start_sec - 45.0)
        prev_lines = [l for l in full_transcript if prev_start <= extract_transcript_times(l)[0] < start_sec]
        previous_context = "\n".join(prev_lines) if prev_lines else "[Beginning of video]"
        
        curr_lines = [l for l in full_transcript if start_sec <= extract_transcript_times(l)[0] <= end_sec]
        chunk_t = "\n".join(curr_lines)
        if not chunk_t.strip(): continue
        
        up_end = end_sec + 45.0
        up_lines = [l for l in full_transcript if end_sec < extract_transcript_times(l)[0] <= up_end]
        upcoming_context = "\n".join(up_lines) if up_lines else "[End of video]"
        
        chunk_o = "\n".join([l for l in full_ocr if prev_start <= extract_ocr_time(l) <= up_end])
        
        print(f"[PASS 1] Scouting {window_label}...")
        
        found = pass_1_scout(chunk_t, chunk_o, window_label, prev_ctx=previous_context, 
                           up_ctx=upcoming_context, min_dur=MIN_CLIP_DURATION, max_dur=MAX_CLIP_DURATION)
        
        if found: 
            print(f"  -> LLM returned {len(found)} potential clips. Validating...")
            for c in found:
                try:
                    s = float(c.get('start', 0))
                    e = float(c.get('end', 0))
                    duration = e - s
                    metrics = c.get('virality_metrics', {})
                    hook = float(metrics.get('hook_strength', 50))
                    payoff = float(metrics.get('payoff_satisfaction', 50))
                    retention = float(metrics.get('retention_potential', 50))
                    llm_score = (hook + payoff + retention) / 3.0
                    
                    if MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION:
                        c['clip_id'] = f"clip_{len(raw_candidate_pool)}" 
                        c['llm_narrative_score'] = round(llm_score, 2)
                        
                        true_virality = calculate_hybrid_virality(s, e, full_transcript, full_ocr, llm_score)
                        c['hybrid_score'] = true_virality
                        
                        padding_data = c.get('padding', {})
                        c['start_padding'] = float(padding_data.get('start_buffer', 0.3))
                        c['end_padding'] = float(padding_data.get('end_buffer', 1.5))
                        
                        raw_candidate_pool.append(c)
                        print(f"  -> [ACCEPTED] {s}s to {e}s | Hybrid Score: {true_virality}")
                    else:
                        print(f"  -> [REJECTED] Clip too short/long: {duration:.1f}s")
                except Exception as ex: 
                    print(f"  -> [REJECTED] Missing or bad timestamps: {ex}")
    
    # 4. Deduplication
    print(f"\n Deduplicating {len(raw_candidate_pool)} raw candidates...")
    clean_candidate_pool = deduplicate_clips(raw_candidate_pool, threshold=0.6)
    print(f"Reduced to {len(clean_candidate_pool)} unique candidates.")
    
    # 5. Pass 2: Editor Final Selection
    print("\n Selecting Final Top Clips...")
    final_clips = pass_2_editor(clean_candidate_pool, max_limit=MAX_TOTAL_CLIPS)
    
    # 6. Export clips
    exported_clips = []
    
    if final_clips:
        with VideoFileClip(VIDEO_PATH) as video:
            for i, selected_clip in enumerate(final_clips):
                try:
                    target_id = selected_clip.get('clip_id')
                    original_clip_data = next((c for c in clean_candidate_pool if c.get('clip_id') == target_id), None)
                    
                    if not original_clip_data:
                        print(f"[ERROR] LLM Hallucinated ID: {target_id}")
                        continue
                    
                    raw_start = float(original_clip_data.get('start', 0))
                    raw_end = float(original_clip_data.get('end', 0))
                    anchor_start = original_clip_data.get('anchor_start_text', '')
                    anchor_end = original_clip_data.get('anchor_end_text', '')
                    
                    if (raw_end - raw_start) < 5.0: raw_end = raw_start + 15.0 
                    
                    true_start = snap_timestamp_to_transcript(anchor_start, raw_start, full_transcript, is_end=False)
                    true_end = snap_timestamp_to_transcript(anchor_end, raw_end, full_transcript, is_end=True)
                    
                    pad_s = max(0.0, min(float(original_clip_data.get('start_padding', 0.3)), 5.0))
                    pad_e = max(0.0, min(float(original_clip_data.get('end_padding', 1.5)), 5.0))
                    
                    s = max(0.0, true_start - pad_s)
                    e = min(video.duration, true_end + pad_e)
                    
                    clip_title = selected_clip.get('title', 'Untitled')
                    file_name = f"viral_clip_{i+1}.mp4"
                    output_fn = os.path.join(OUTPUT_DIR, file_name)
                    
                    print(f"\n[EXPORT] {file_name} ({format_time(s)} - {format_time(e)})")
                    print(f"        Title: '{clip_title}'")
                    
                    # Version-agnostic clipping
                    if hasattr(video, "subclipped"):
                        subclip = video.subclipped(s, e)
                    else:
                        subclip = video.subclip(s, e)
                    subclip.write_videofile(
                        output_fn, 
                        codec="libx264", 
                        audio_codec="aac", 
                        threads=4, 
                        preset="superfast", 
                        logger=None
                    )
                    
                    exported_clips.append({
                        "file_name": file_name,
                        "file_path": output_fn,
                        "title": clip_title,
                        "start_time": s,
                        "end_time": e,
                        "duration": e - s
                    })
                    
                except Exception as ex:
                    print(f"[ERROR] Skipping clip {i+1}: {ex}")
    
    # Save metadata
    if exported_clips:
        metadata_path = os.path.join(OUTPUT_DIR, "clips_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(exported_clips, f, indent=4)
        print(f"\n Saved clip metadata to: {metadata_path}")
    
    return exported_clips

# --- UTILS ---
def snap_timestamp_to_transcript(anchor_text, target_time, full_transcript, is_end=False):
    if not anchor_text or anchor_text in ["first four words exactly", "last four words exactly"]:
        return target_time 

    anchor_clean = re.sub(r'[^\w\s]', '', anchor_text.lower().strip())
    best_match_time = target_time
    highest_ratio = 0.0

    for line in full_transcript:
        start_t, end_t = extract_transcript_times(line)
        if start_t == -1.0: continue
        
        if abs(start_t - target_time) < 15.0:
            line_text = re.sub(r'\[.*?\] \(.*?\) ', '', line) 
            line_clean = re.sub(r'[^\w\s]', '', line_text.lower().strip())
            
            ratio = difflib.SequenceMatcher(None, anchor_clean, line_clean).ratio()
            if anchor_clean in line_clean: ratio = 1.0 

            if ratio > highest_ratio and ratio > 0.4:
                highest_ratio = ratio
                best_match_time = end_t if is_end else start_t

    if highest_ratio > 0.4:
        print(f"      [SNAP] Adjusted LLM time {target_time}s -> {best_match_time}s based on text: '{anchor_text}'")
        return best_match_time
    
    return target_time 

def get_safe_json(raw_response):
    try:
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        clean_text = re.sub(r'```json|```', '', clean_text, flags=re.IGNORECASE).strip()
        match = re.search(r'\[.*\]', clean_text, re.DOTALL)
        if match: clean_text = match.group(0)
        return json_repair.loads(clean_text)
    except: return None

def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"

# --- SEMANTIC CHUNKING UTILS ---
def extract_transcript_times(line):
    match = re.search(r'\[([\d\.]+)s - ([\d\.]+)s\]', line)
    if match: return float(match.group(1)), float(match.group(2))
    return -1.0, -1.0

def extract_ocr_time(line):
    match = re.search(r'\[([\d\.]+)s\]', line)
    if match: return float(match.group(1))
    return -1.0

def generate_semantic_chapters(transcript, ocr_data, min_length=45, max_length=180, pause_threshold=2.0):
    chapters = []
    current_start = 0.0
    last_end = 0.0
    ocr_times = [extract_ocr_time(l) for l in ocr_data]
    
    for line in transcript:
        start, end = extract_transcript_times(line)
        if start == -1.0: continue
        
        if current_start == 0.0 and len(chapters) == 0:
            current_start = start
            
        is_long_pause = (start - last_end) >= pause_threshold and last_end > 0
        is_scene_change = any((last_end - 1.0) <= t <= (start + 1.0) for t in ocr_times)
        is_too_long = (end - current_start) >= max_length
        is_long_enough = (last_end - current_start) >= min_length
        
        if (is_long_pause or is_scene_change or is_too_long) and is_long_enough:
            chapters.append((current_start, last_end))
            current_start = start 
            
        last_end = end
        
    if (last_end - current_start) > 10:
        chapters.append((current_start, last_end))
        
    return chapters

# --- HYBRID SCORING ---
def calculate_hybrid_virality(start, end, full_transcript, full_ocr, llm_score):
    duration = end - start
    if duration <= 0: return llm_score
    
    ocr_in_window = [l for l in full_ocr if start <= extract_ocr_time(l) <= end]
    expected_changes = duration / 4.0 
    visual_score = min((len(ocr_in_window) / max(expected_changes, 1)) * 100, 100)

    transcript_in_window = [l for l in full_transcript if start <= extract_transcript_times(l)[0] <= end]
    total_lines = max(len(transcript_in_window), 1)
    
    loud_count = sum(1 for l in transcript_in_window if "Vol: Loud" in l)
    fast_count = sum(1 for l in transcript_in_window if "Pace: Fast" in l)
    
    intensity_ratio = (loud_count + (fast_count * 1.5)) / total_lines
    audio_score = min(intensity_ratio * 100, 100)
    
    final_score = (float(llm_score) * 0.70) + (visual_score * 0.15) + (audio_score * 0.15)
    return round(final_score, 2)

# --- DEDUPLICATION LOGIC ---
def calculate_iou(clip1, clip2):
    start1, end1 = float(clip1.get('start', 0)), float(clip1.get('end', 0))
    start2, end2 = float(clip2.get('start', 0)), float(clip2.get('end', 0))
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0

def deduplicate_clips(candidates, threshold=0.6):
    sorted_cands = sorted(candidates, key=lambda x: x.get('hybrid_score', 0), reverse=True)
    unique_clips = []
    for c in sorted_cands:
        is_duplicate = False
        for u in unique_clips:
            if calculate_iou(c, u) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_clips.append(c)
    return unique_clips

# --- EXTRACTION WITH MULTI-MODAL SIGNALS ---
def get_transcript_with_signals(video_path, model):
    print(f"[LOG] Transcribing and analyzing audio signals...")
    segments, _ = model.transcribe(video_path, vad_filter=True, word_timestamps=True)
    
    video = VideoFileClip(video_path)
    audio_clip = video.audio
    transcript_data = []
    
    for s in segments:
        duration = s.end - s.start
        word_count = len(s.text.split())
        wps = word_count / duration if duration > 0 else 0
        pace = "Fast" if wps > 3.5 else "Slow" if wps < 1.5 else "Normal"
        
        try:
            end_time = min(s.end, audio_clip.duration)
            
            # Version-agnostic audio clipping
            if hasattr(audio_clip, "subclipped"):
                arr = audio_clip.subclipped(s.start, end_time).to_soundarray(fps=8000)
            else:
                arr = audio_clip.subclip(s.start, end_time).to_soundarray(fps=8000)
            if arr is not None and len(arr) > 0:
                rms = np.sqrt(np.mean(arr**2))
                loudness = "Loud" if rms > 0.15 else "Quiet" if rms < 0.03 else "Normal"
            else: loudness = "Normal"
        except: loudness = "Normal"
            
        line = f"[{s.start:.1f}s - {s.end:.1f}s] (Vol: {loudness}, Pace: {pace}) {s.text.strip()}"
        transcript_data.append(line)
        
    video.close()
    return transcript_data

def get_ocr_list(video_path, reader):
    print(f"[LOG] Scanning OCR...")
    cap = cv2.VideoCapture(video_path)
    ocr_data, last_text = [], ""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_interval = int(fps * 1) 
    
    pbar = tqdm(total=total_frames)
    frame_count = 0
    while True:
        ret = cap.grab()
        if not ret: break
        
        if frame_count % skip_interval == 0:
            ret, frame = cap.retrieve()
            if ret:
                text = " ".join(reader.readtext(frame, detail=0)).strip()
                if len(text) > 4 and difflib.SequenceMatcher(None, last_text, text).ratio() < 0.8:
                    ocr_data.append(f"[{frame_count/fps:.1f}s] SCREEN TEXT CHANGED TO: {text}")
                    last_text = text
                    
        frame_count += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    return ocr_data

# --- PASS 1: THE SCOUT ---
def pass_1_scout(transcript_chunk, ocr_chunk, window_label, prev_ctx="", up_ctx="", min_dur=45.0, max_dur=90.0):
    system_instruction = (
        "You are an Elite Narrative Architect and Viral Content Editor. Your mission is to extract 'Golden Moments'—high-retention, high-density short-form stories.\n\n"
        "### PHASE 1: THE VIRAL ANATOMY AUDIT\n"
        "1. THE HOOK (0-3s): Must be a 'Pattern Interrupt'—a bold claim, high energy, or a visual shift.\n"
        "2. INFORMATION DENSITY: Prioritize high-pace, high-energy, highly emotional, or heavily debatable dialogue.\n"
        "3. THE PAYOFF: The clip MUST end with a satisfying punchline, reveal, or reaction. Never cut off the 'Result'.\n\n"
        "### PHASE 2: DURATION STRICTNESS (CRITICAL RULE)\n"
        f"4. DURATION: Clips MUST be strictly between {min_dur} and {max_dur} seconds.\n"
        f"5. THE 'CONTEXT PADDING' RULE: If a funny punchline or reaction is only 10 seconds long, you MUST include the context leading up to it to hit the {min_dur}-second minimum.\n\n"
        "### PHASE 3: CONTEXTUAL BLEED OVER & DEDUPLICATION (NEW)\n"
        "6. THE BLEED RULE: You are provided with 'PREVIOUS CONTEXT' and 'UPCOMING CONTEXT'. If a story arc or joke begins in the previous context or bleeds into the upcoming context, you ARE ALLOWED to use those timestamps to ensure the clip is complete. Our backend system will automatically deduplicate any overlapping clips.\n\n"
        "### PHASE 4: TEMPORAL PRECISION & ANCHORS\n"
        "7. SAFE ENTRY: Start 0.3s before the first word to catch the breath.\n"
        "8. OCR-SNAP EXIT: Cross-reference OCR. If a 'Scene Change' happens within 1.0s of the final word, SNAP the end_padding to that exact transition.\n"
        "9. ANCHOR TEXT: You must provide EXACTLY the first 4 words and EXACTLY the last 4 words of the clip.\n\n"
        "### PHASE 5: DYNAMIC EXTRACTION (QUALITY OVER QUANTITY)\n"
        "10. EXTRACTION LIMIT: There is NO limit. If the current transcript has 4 highly debatable moments, extract all 4. \n"
        "11. THE 'SKIP' RULE: If the transcript is boring, logistical, or lacks a satisfying payoff, DO NOT force a clip. Output an empty JSON array [].\n\n"
        "### PHASE 6: CHAIN OF THOUGHT\n"
        f"Your <think> block MUST explicitly calculate: End_Time - Start_Time = Duration. If the Duration is less than {min_dur} seconds, you must rewrite the clip to include more setup.\n"
        "Output ONLY the <think> block followed immediately by the raw JSON array. Do not output any other text."
    )

    user_prompt = (
        f"DATASET ANALYSIS ({window_label}):\n\n"
        f"--- PREVIOUS CONTEXT (What happened just before this chunk) ---\n{prev_ctx}\n\n"
        f"--- CURRENT CHAPTER (Main focus area) ---\n{transcript_chunk}\n\n"
        f"--- UPCOMING CONTEXT (What happens right after this chunk) ---\n{up_ctx}\n\n"
        f"--- FULL CONTEXT OCR (Visual Scene Transitions) ---\n{ocr_chunk}\n\n"
        "INSTRUCTIONS:\n"
        "Extract ALL standalone 'Golden Moments'. You may pull timestamps from the Previous or Upcoming Contexts if the narrative requires it. Return the JSON using the following schema (return [] if no moments meet the high standards):\n"
        "[\n"
        "  {\n"
        "    \"clip_title\": \"Hook-driven title for the clip\",\n"
        "    \"start\": 0.0,\n"
        "    \"end\": 0.0,\n"
        f"    \"duration_check\": \"Calculate: End - Start. State the total seconds. MUST be {min_dur}-{max_dur}s.\",\n"
        "    \"anchor_start_text\": \"first four words exactly\",\n"
        "    \"anchor_end_text\": \"last four words exactly\",\n"
        "    \"padding\": {\"start_buffer\": 0.3, \"end_buffer\": 1.5},\n"
        "    \"temporal_math\": \"Detailed calculation of the padding chosen.\",\n"
        "    \"context_check\": \"Explanation of why this clip makes sense standalone.\",\n"
        "    \"virality_metrics\": {\"hook_strength\": 95, \"payoff_satisfaction\": 90, \"retention_potential\": 85},\n"
        "    \"reasoning\": \"Why this moment will perform well on TikTok/Reels.\"\n"
        "  }\n"
        "]"
    )
    
    response = client.chat.completions.create(
        model=SCOUT_MODEL, 
        messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}],
        temperature=0.2 
    )
    
    raw_output = response.choices[0].message.content
    print(f"\n--- RAW LLM OUTPUT ---\n{raw_output}\n----------------------\n")
    return get_safe_json(raw_output)

# --- PASS 2: THE EDITOR (BATCHED) ---
def pass_2_editor(candidate_pool, max_limit=5):
    if not candidate_pool:
        print("[ERROR] Candidate pool is empty. Skipping Pass 2.")
        return []

    system_instruction = (
        "You are the Lead Short-Form Content Strategist for a massive YouTube channel. Your mission is to curate an elite 'Viral Batch' from a pool of candidates.\n\n"
        "### STRATEGIC SELECTION RULES:\n"
        "1. THE DIVERSITY MANDATE: Do not pick multiple clips covering the exact same story beat. Curate a mix of proven 'Viral Archetypes':\n"
        "   - The 'High-Stakes Challenge' (Immediate danger/action)\n"
        "   - The 'Mind-Blowing Fact' (High educational/insight value)\n"
        "   - The 'Hilarious/Raw Reaction' (Pure emotion/humor)\n"
        "   - The 'Hot Take / Debate' (Highly controversial, prompts heavy comments)\n"
        "   - The 'Satisfying Process' (A clear beginning, middle, and result)\n"
        "2. THE ENGAGEMENT TRIGGER: Prioritize clips that force a user behavior. Will they share this with a friend? Will they angrily comment to disagree? Will they re-watch it because it loops perfectly?\n"
        "3. LOGICAL COMPLETENESS: Reject any clip that feels like 'the middle of a thought'.\n\n"
        "### OUTPUT CONSTRAINTS (QUALITY OVER QUANTITY):\n"
        f"- THE STRICT GATEKEEPER RULE: There is NO quota. You may select 1, 10, or 0 clips. ONLY select candidates that have a 90+ retention potential and a clear engagement trigger. Discard the rest. Never output more than {max_limit} total.\n"
        "- Titles must be punchy 'Hook Text' designed to be plastered on the center of the video (Max 6 words).\n"
        "- Output ONLY raw JSON. No markdown blocks, no preamble."
    )
    
    final_elite_clips = []
    
    # Process in batches of 15 to protect context limits on large videos
    batch_size = 15
    for i in range(0, len(candidate_pool), batch_size):
        batch = candidate_pool[i:i + batch_size]
        
        slim_pool = []
        for c in batch: 
            slim_pool.append({
                "clip_id": c.get("clip_id"),
                "clip_title": c.get("clip_title", "No title provided"), 
                "reasoning": c.get("reasoning", "No reasoning provided"),
                "narrative_score": c.get("llm_narrative_score", 0),
                "energy_score": c.get("hybrid_score", 0) 
            })

        user_prompt = (
            f"CANDIDATE POOL (Batch {i//batch_size + 1}):\n{json.dumps(slim_pool, indent=2)}\n\n"
            "TASK: Review this batch and select ONLY the elite candidates that are guaranteed to drive engagement.\n\n"
            "REQUIRED JSON FORMAT:\n"
            "[\n"
            "  {\n"
            "    \"clip_id\": \"string\",\n"
            "    \"title\": \"CATCHY OVERLAY TITLE\",\n"
            "    \"viral_archetype\": \"Which of the 5 archetypes this fits\",\n"
            "    \"engagement_trigger\": \"Why will people comment on or share this specific clip?\",\n"
            "    \"selection_reason\": \"Why this survived the strict quality filter\"\n"
            "  }\n"
            "]"
        )
        
        print(f"\n[LOG] Swapping models... Loading Editor Model for Batch {i//batch_size + 1}...")
        response = client.chat.completions.create(
            model=EDITOR_MODEL, 
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        batch_results = get_safe_json(response.choices[0].message.content)
        if batch_results:
            final_elite_clips.extend(batch_results)
            
    if len(final_elite_clips) > max_limit:
        print(f"[LOG] LLM selected {len(final_elite_clips)} clips. Trimming down to the maximum limit of {max_limit}.")
        return final_elite_clips[:max_limit]
        
    return final_elite_clips

# --- MAIN RUNNER ---
if __name__ == "__main__":
    # Ensure LOCAL_BIN_DIR is available
    original_path = os.environ.get("PATH", "")
    if LOCAL_BIN_DIR not in original_path:
        os.environ["PATH"] = LOCAL_BIN_DIR + os.pathsep + original_path
        print(f" Added {LOCAL_BIN_DIR} to PATH for ClipperM module")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    with VideoFileClip(VIDEO_PATH) as video: 
        total_duration = video.duration

    reader = easyocr.Reader(['en'], gpu=True)
    whisper_model = WhisperModel("small", device="cuda", compute_type="float16")

    # 1. Extraction
    full_transcript = get_transcript_with_signals(VIDEO_PATH, whisper_model)
    full_ocr = get_ocr_list(VIDEO_PATH, reader)

    # 2. PASS 1: Semantic Scouting
    print("\n[LOG] Analyzing video for semantic chapter breaks...")
    semantic_chapters = generate_semantic_chapters(full_transcript, full_ocr)
    print(f"[LOG] Found {len(semantic_chapters)} logical chapters.")

    raw_candidate_pool = []

    for i, (start_sec, end_sec) in enumerate(semantic_chapters):
        window_label = f"Chapter {i+1} ({format_time(start_sec)} to {format_time(end_sec)})"
        
        # --- NEW: EXACT PREVIOUS, CURRENT, AND UPCOMING CONTEXT BLOCKS ---
        
        # 1. Previous Context (45 seconds before chapter starts)
        prev_start = max(0, start_sec - 45.0)
        prev_lines = [l for l in full_transcript if prev_start <= extract_transcript_times(l)[0] < start_sec]
        previous_context = "\n".join(prev_lines) if prev_lines else "[Beginning of video]"

        # 2. Current Chapter
        curr_lines = [l for l in full_transcript if start_sec <= extract_transcript_times(l)[0] <= end_sec]
        chunk_t = "\n".join(curr_lines)
        if not chunk_t.strip(): continue

        # 3. Upcoming Context (45 seconds after chapter ends)
        up_end = end_sec + 45.0
        up_lines = [l for l in full_transcript if end_sec < extract_transcript_times(l)[0] <= up_end]
        upcoming_context = "\n".join(up_lines) if up_lines else "[End of video]"
        
        # 4. Total OCR Data spanning all three windows
        chunk_o = "\n".join([l for l in full_ocr if prev_start <= extract_ocr_time(l) <= up_end])

        print(f"[PASS 1] Scouting {window_label}...")
        
        # Inject our precise contexts and min/max variables into Pass 1
        found = pass_1_scout(chunk_t, chunk_o, window_label, prev_ctx=previous_context, up_ctx=upcoming_context, min_dur=MIN_CLIP_DURATION, max_dur=MAX_CLIP_DURATION)
        
        if found: 
            print(f"  -> LLM returned {len(found)} potential clips. Validating...")
            for c in found:
                try:
                    s = float(c.get('start', 0))
                    e = float(c.get('end', 0))
                    duration = e - s
                    metrics = c.get('virality_metrics', {})
                    hook = float(metrics.get('hook_strength', 50))
                    payoff = float(metrics.get('payoff_satisfaction', 50))
                    retention = float(metrics.get('retention_potential', 50))
                    llm_score = (hook + payoff + retention) / 3.0
                    
                    if MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION:
                        c['clip_id'] = f"clip_{len(raw_candidate_pool)}" 
                        c['llm_narrative_score'] = round(llm_score, 2)
                        
                        true_virality = calculate_hybrid_virality(s, e, full_transcript, full_ocr, llm_score)
                        c['hybrid_score'] = true_virality

                        padding_data = c.get('padding', {})
                        c['start_padding'] = float(padding_data.get('start_buffer', 0.3))
                        c['end_padding'] = float(padding_data.get('end_buffer', 1.5))
                        
                        raw_candidate_pool.append(c)
                        print(f"  -> [ACCEPTED] {s}s to {e}s | Hybrid Score: {true_virality} (Narrative: {c['llm_narrative_score']}) | Pad: +{c['start_padding']}s / +{c['end_padding']}s")
                    else:
                        print(f"  -> [REJECTED] Clip too short/long: {s}s to {e}s (Duration: {round(duration, 1)}s)")
                except Exception as ex: 
                    print(f"  -> [REJECTED] Missing or bad timestamps in JSON: {ex}")

    # 3. DEDUPLICATION
    print(f"\n[LOG] Deduplicating {len(raw_candidate_pool)} raw candidates...")
    clean_candidate_pool = deduplicate_clips(raw_candidate_pool, threshold=0.6)
    print(f"[LOG] Reduced to {len(clean_candidate_pool)} unique candidates after IoU overlap check.")

    # 4. PASS 2: Editor Final Selection
    print("\n[PASS 2] Selecting Final Top Clips...")
    final_clips = pass_2_editor(clean_candidate_pool, max_limit=MAX_TOTAL_CLIPS)

    # 5. Export Standard Trims & Meta Data (Using MoviePy as requested)
    exported_clips_metadata = []

    if final_clips:
        with VideoFileClip(VIDEO_PATH) as video:
            for i, selected_clip in enumerate(final_clips):
                try:
                    target_id = selected_clip.get('clip_id')
                    original_clip_data = next((c for c in clean_candidate_pool if c.get('clip_id') == target_id), None)
                    
                    if not original_clip_data:
                        print(f"[ERROR] LLM Hallucinated ID: {target_id}")
                        continue

                    # Grab the raw timestamps and anchor text
                    raw_start = float(original_clip_data.get('start', 0))
                    raw_end = float(original_clip_data.get('end', 0))
                    anchor_start = original_clip_data.get('anchor_start_text', '')
                    anchor_end = original_clip_data.get('anchor_end_text', '')
                    
                    if (raw_end - raw_start) < 5.0: raw_end = raw_start + 15.0 
                    
                    # --- SNAP TO TRUE WHISPER TIMESTAMPS ---
                    true_start = snap_timestamp_to_transcript(anchor_start, raw_start, full_transcript, is_end=False)
                    true_end = snap_timestamp_to_transcript(anchor_end, raw_end, full_transcript, is_end=True)
                    
                    pad_s = max(0.0, min(float(original_clip_data.get('start_padding', 0.3)), 5.0))
                    pad_e = max(0.0, min(float(original_clip_data.get('end_padding', 1.5)), 5.0))
                    
                    s = max(0.0, true_start - pad_s)
                    e = min(video.duration, true_end + pad_e)
                    
                    # --- Console Logging ---
                    clip_title = selected_clip.get('title', 'Untitled')
                    narrative_score = original_clip_data.get('llm_narrative_score', 0)
                    hybrid_score = original_clip_data.get('hybrid_score', 0)
                    file_name = f"viral_clip_{i+1}.mp4"

                    print(f"\n[EXPORT] {file_name} ({format_time(s)} - {format_time(e)})")
                    print(f"         Title: '{clip_title}'")
                    print(f"         Engagement Trigger: '{selected_clip.get('engagement_trigger', 'N/A')}'")
                    print(f"         Scores - Narrative: {narrative_score} | Energy/Hybrid: {hybrid_score}")
                    
                    output_fn = os.path.join(OUTPUT_DIR, file_name)
                    
                    # Create subclip and render fast with threads
                    if hasattr(video, "subclipped"):
                        subclip = video.subclipped(s, e)
                    else:
                        subclip = video.subclip(s, e)
                    subclip.write_videofile(
                        output_fn, 
                        codec="libx264", 
                        audio_codec="aac", 
                        threads=4, 
                        preset="superfast", 
                        logger=None
                    )
                    
                    # Add successful export to our JSON collector
                    clip_metadata = {
                        "file_name": file_name,
                        "title": clip_title,
                        "viral_archetype": selected_clip.get("viral_archetype", ""),
                        "engagement_trigger": selected_clip.get("engagement_trigger", ""),
                        "selection_reason": selected_clip.get("selection_reason", ""),
                        "start_time_seconds": round(s, 2),
                        "end_time_seconds": round(e, 2),
                        "duration_seconds": round(e - s, 2),
                        "narrative_score": narrative_score,
                        "energy_score": hybrid_score
                    }
                    exported_clips_metadata.append(clip_metadata)
                    
                except Exception as ex:
                    print(f"[ERROR] Skipping clip {i+1}: {ex}")

        # Dump all metadata to a JSON file
        if exported_clips_metadata:
            metadata_path = os.path.join(OUTPUT_DIR, "clips_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(exported_clips_metadata, f, indent=4)
            print(f"\n[LOG] Success! Saved complete clip metadata to: {metadata_path}")

    else:
        print("[LOG] No final clips to export.")