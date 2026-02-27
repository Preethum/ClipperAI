"""
Podcast scenario configuration and interface.
This file uses global functions from global_functions.py and holds all podcast-specific configurations.
"""

import os
import json
from global_functions import run_complete_pipeline

def get_podcast_config():
    """Get default configuration for podcast processing."""
    return {
        # Input/Output paths
        "input_video": "source.mp4",
        "base_output_dir": "output",
        
        # Module configuration
        "modules": {
            "clipper": {
                "enabled": True,
                "min_clip_duration": 45.0,
                "max_clip_duration": 90.0,
                "max_total_clips": 10,
                
                # ClipperM accepted parameters
                "scout_model": "deepseek-r1-distill-qwen-32b",
                "editor_model": "google/gemma-3-27b",
                "lm_studio_url": "http://localhost:1234/v1",
                
                # Vision and OCR Settings
                "enable_ocr": False,
                "enable_vision": True,
                "vision_model": "qwen/qwen3-vl-30b",
                "vision_interval": 2.0,
                "vision_concurrency": 5
                
                # Deduplication settings
                "deduplication_threshold": 0.5,  # Less aggressive for podcast content
                
                # Viral archetypes for strategic selection
                "viral_archetypes": [
                    "High-Stakes Challenge",
                    "Mind-Blowing Fact", 
                    "Hilarious/Raw Reaction",
                    "Hot Take / Debate",
                    "Satisfying Process"
                ],
                
                # Custom LLM prompts for this scenario
                "scout_system_instruction": (
                    "You are an Elite Narrative Architect and Viral Content Editor. Your mission is to extract 'Golden Moments'—high-retention, high-density short-form stories.\n\n"
                    "### PHASE 1: THE VIRAL ANATOMY AUDIT\n"
                    "1. THE HOOK (0-3s): Must be a 'Pattern Interrupt'—a bold claim, high energy, or a visual shift.\n"
                    "2. INFORMATION DENSITY: Prioritize high-pace, high-energy, highly emotional, or heavily debatable dialogue.\n"
                    "3. THE PAYOFF: The clip MUST end with a satisfying punchline, reveal, or reaction. Never cut off the 'Result'.\n\n"
                    "### PHASE 2: DURATION STRICTNESS (CRITICAL RULE)\n"
                    "4. DURATION: Clips MUST be strictly between {min_dur} and {max_dur} seconds.\n"
                    "5. THE 'CONTEXT PADDING' RULE: If a funny punchline or reaction is only 10 seconds long, you MUST include the context leading up to it to hit the {min_dur}-second minimum.\n\n"
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
                    "Your </think> block MUST explicitly calculate: End_Time - Start_Time = Duration. If the Duration is less than {min_dur} seconds, you must rewrite the clip to include more setup.\n"
                    "Output ONLY the </think> block followed immediately by the raw JSON array. Do not output any other text.\n\n"
                    "### PHASE 7: BREVITY RULE (CRITICAL)\n"
                    "Keep ALL text fields concise. The 'reasoning' field MUST be 10 sentences maximum. The 'context_check' and 'temporal_math' fields MUST each be 3 sentences maximum. Do NOT write paragraphs."
                ),
                
                "scout_user_prompt": (
                    "DATASET ANALYSIS ({window_label}):\n\n"
                    "--- PREVIOUS CONTEXT (What happened just before this chunk) ---{prev_ctx}\n\n"
                    "--- CURRENT CHAPTER (Main focus area) ---{transcript_chunk}\n\n"
                    "--- UPCOMING CONTEXT (What happens right after this chunk) ---{up_ctx}\n\n"
                    "--- FULL CONTEXT OCR (Visual Scene Transitions) ---{ocr_chunk}\n\n"
                    "--- FULL VISUAL CONTEXT (Keyframe Descriptions) ---{vision_chunk}\n\n"
                    "INSTRUCTIONS:\n"
                    "Extract ALL standalone 'Golden Moments'. You may pull timestamps from the Previous or Upcoming Contexts if the narrative requires it. Return the JSON using the following schema (return [] if no moments meet the high standards):\n"
                    "[\n"
                    "  {{\n" # <-- DOUBLED HERE
                    "    \"clip_title\": \"Hook-driven title for the clip\",\n"
                    "    \"start\": 0.0,\n"
                    "    \"end\": 0.0,\n"
                    "    \"duration_check\": \"Calculate: End - Start. State the total seconds. MUST be {min_dur}-{max_dur}s.\",\n"
                    "    \"anchor_start_text\": \"first four words exactly\",\n"
                    "    \"anchor_end_text\": \"last four words exactly\",\n"
                    "    \"padding\": {{\"start_buffer\": 0.3, \"end_buffer\": 1.5}},\n" # <-- DOUBLED HERE
                    "    \"temporal_math\": \"Detailed calculation of the padding chosen.\",\n"
                    "    \"context_check\": \"Explanation of why this clip makes sense standalone.\",\n"
                    "    \"virality_metrics\": {{\"hook_strength\": 95, \"payoff_satisfaction\": 90, \"retention_potential\": 85}},\n" # <-- DOUBLED HERE
                    "    \"reasoning\": \"Why this moment will perform well on TikTok/Reels.\"\n"
                    "  }}\n" # <-- DOUBLED HERE
                    "]"
                ),
                
                "editor_system_instruction": (
                    "You are a Lead Short-Form Content Strategist for a massive YouTube channel. Your mission is to curate an elite 'Viral Batch' from a pool of candidates.\n\n"
                    "### STRATEGIC SELECTION RULES:\n"
                    "1. THE DIVERSITY MANDATE: Do not pick multiple clips covering the exact same story beat. Curate a mix of proven 'Viral Archetypes':\n"
                    "{archetypes_list}"
                    "2. THE ENGAGEMENT TRIGGER: Prioritize clips that force a user behavior. Will they share this with a friend? Will they angrily comment to disagree? Will they re-watch it because it loops perfectly?\n"
                    "3. LOGICAL COMPLETENESS: Reject any clip that feels like 'the middle of a thought'.\n\n"
                    "### OUTPUT CONSTRAINTS (QUALITY OVER QUANTITY):\n"
                    "- THE BALANCED GATEKEEPER RULE: Select clips with 75+ retention potential and a clear engagement trigger. Aim for quality but be more generous to provide variety. Never output more than {max_limit} total.\n"
                    "- Titles must be punchy 'Hook Text' designed to be plastered on center of the video (Max 6 words).\n"
                    "- Output ONLY raw JSON. No markdown blocks, no preamble."
                ),
                
                "editor_user_prompt": (
                    "CANDIDATE POOL (Batch {batch_num}):\n{batch_json}\n\n"
                    "TASK: Review this batch and select ONLY the elite candidates that are guaranteed to drive engagement.\n\n"
                    "REQUIRED JSON FORMAT:\n"
                    "[\n"
                    "  {{\n"
                    "    \"clip_id\": \"string\",\n"
                    "    \"title\": \"CATCHY OVERLAY TITLE\",\n"
                    "    \"viral_archetype\": \"Which of the 5 archetypes this fits\",\n"
                    "    \"engagement_trigger\": \"Why will people comment on or share this specific clip?\",\n"
                    "    \"selection_reason\": \"Why this survived the strict quality filter\"\n"
                    "  }}\n"
                    "]"
                )
            },
            "cropper": {
                "enabled": True,
                
                # CropperM accepted parameters
                "ratio": "9:16",
                "quality": "balanced",
                "crf": None,
                "preset": None,
                "plan_only": False,
                "frame_skip": 0,
                "downscale": 0,
                "encoder": "auto"
            },
            "subs": {
                "enabled": True,
                
                # SubsM accepted parameters
                "template": "templates/hype",
                "vertical_align_offset": 0.70,
                "max_width_ratio": 0.9,
                "max_lines": 2
            }
        },
        
        # Processing options
        "cleanup_temp": True,
        "preserve_intermediate": False,
        
        # Global Scenario Settings
        "scenario_name": "podcast",
        "target_platform": "youtube_shorts",
        "content_style": "educational_entertainment"
    }

def main(config_file=None):
    """
    Main function to run the podcast scenario.
    
    Args:
        config_file (str): Path to JSON configuration file. If None, uses default config.
    """
    # Load configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f" Loaded configuration from: {config_file}")
    else:
        config = get_podcast_config()
        print(" Using default podcast configuration")
    
    print(f" Input video: {config.get('input_video', 'Not specified')}")
    
    # Run complete pipeline using global functions
    return run_complete_pipeline(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Podcast Video Processing Pipeline")
    parser.add_argument('--config', type=str, help="Path to JSON configuration file")
    parser.add_argument('--input', type=str, help="Input video file path")
    parser.add_argument('--output', type=str, help="Output directory path")
    parser.add_argument('--clips', type=int, default=10, help="Maximum number of clips to generate")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_podcast_config()
        
        # Override with command line arguments
        if args.input:
            config["input_video"] = args.input
        if args.output:
            config["base_output_dir"] = args.output
        config["modules"]["clipper"]["max_total_clips"] = args.clips
    
    print(f"📹 Input video: {config.get('input_video', 'Not specified')}")
    
    # Run scenario using global functions
    final_clips = run_complete_pipeline(config)
    
    print(f"\n✅ Processing complete! Generated {len(final_clips)} final videos.")
