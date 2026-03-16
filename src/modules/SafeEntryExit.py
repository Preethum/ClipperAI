"""
SafeEntryExit.py - Audio + VLM-powered safe entry/exit detection for gaming clips.
Finds perfect cut points using audio intensity analysis first, then VLM as fallback.
"""

import json
import os
import cv2
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

# Allow type hints without circular imports
if TYPE_CHECKING:
    from AudioAnalyzerM import AudioAnalyzer

# Try to import LM Studio SDK
try:
    from lmstudio import Client
    LMS_AVAILABLE = True
except ImportError:
    LMS_AVAILABLE = False
    print("⚠️  LM Studio SDK not available, using fallback mode")


class SafeEntryExitDetector:
    """Finds safe entry/exit points for gaming clips.
    
    Uses a two-tier approach:
      1. Audio analysis (fast, GPU-free): finds combat onset/offset via RMS energy.
      2. VLM batched analysis (fallback): 25-batch rolling context frame analysis.
    """
    
    def __init__(self, video_path: str, vlm_url: str = "http://localhost:1234", 
                 vlm_model: str = "google/gemma-3-27b", fps: int = 10,
                 audio_analyzer: Optional["AudioAnalyzer"] = None):
        """
        Args:
            video_path:      Path to the source video.
            vlm_url:         LM Studio base URL.
            vlm_model:       LM Studio model ID.
            fps:             Frame sampling rate for VLM analysis.
            audio_analyzer:  Optional AudioAnalyzer instance. When provided, audio
                             analysis runs first. VLM only used as fallback.
        """
        self.video_path = video_path
        self.vlm_url = vlm_url.rstrip('/')
        self.vlm_model = vlm_model
        self.fps = fps
        self.cap = None
        self.audio_analyzer = audio_analyzer
        
        if audio_analyzer is not None:
            print("[Audio] Audio-first mode: will use AudioAnalyzer for entry/exit, VLM as fallback")
        
        # Initialize LM Studio client if available
        self.lm_client = None
        if LMS_AVAILABLE:
            try:
                # Extract host from URL (remove protocol and /v1)
                import re
                host = re.sub(r'^(?:https?|wss?)://', '', self.vlm_url).rstrip('/').removesuffix('/v1')
                self.lm_client = Client(api_host=host)
                # Quick connectivity test
                print(f"✅ LM Studio client initialized (host: {host})")
            except Exception as e:
                print(f"⚠️  LM Studio SDK failed: {e}")
                print(f"   Will use HTTP API fallback instead")
                self.lm_client = None
        
    def _open_video(self):
        """Open video capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
    
    def _extract_frames(self, start_time: float, duration: float) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video for VLM analysis.
        
        Args:
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            List of (timestamp, frame) tuples
        """
        self._open_video()
        
        frames = []
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time * video_fps)
        end_frame = min(int((start_time + duration) * video_fps), total_frames)
        
        # Calculate frame step for target fps
        frame_step = max(1, int(video_fps / self.fps))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = current_frame / video_fps
            frames.append((timestamp, frame.copy()))
            
            # Skip frames to achieve target fps
            for _ in range(frame_step - 1):
                current_frame += 1
                self.cap.grab()
                if current_frame >= end_frame:
                    break
            
            current_frame += 1
        
        return frames
    
    def _analyze_frames_with_vlm(self, frames: List[Tuple[float, np.ndarray]], 
                                analysis_type: str, event_context: str = "",
                                analysis_start: float = 0, clip_start_time: float = 0, 
                                clip_end_time: float = 0, buffer_seconds: float = 60) -> Dict:
        """
        Send frames to LM Studio VLM for safe entry/exit analysis.
        Uses batched processing: all frames are sent in batches, then results are aggregated.
        
        Args:
            frames: List of (timestamp, frame) tuples
            analysis_type: "entry" or "exit"
            
        Returns:
            Analysis results with safe timestamps
        """
        if not frames:
            return {"safe_timestamp": None, "confidence": 0.0}
        
        # Build the prompt
        if analysis_type == "entry":
            prompt = f"""You are analyzing a gaming clip for the perfect entry point. 

CONTEXT: Frames from a {buffer_seconds:.0f}s window starting at {analysis_start:.0f}s. The main action starts at {clip_start_time:.0f}s.{event_context}

TASK: Find the FIGHT START point - when combat action begins and viewers should start watching.

Return a timestamp between {analysis_start:.0f} and {clip_start_time:.0f}.

Look for: first shot fired, weapon drawn, enemy appears, health/shield changes, combat-ready movement.

Return only the timestamp in seconds like: {clip_start_time - 5.0:.0f}"""
        else:  # exit
            prompt = f"""You are analyzing a gaming clip for the perfect exit point.

CONTEXT: Frames from a {buffer_seconds:.0f}s window starting at {analysis_start:.0f}s. The main action ends at {clip_end_time:.0f}s.{event_context}

TASK: Find the FIGHT END point - when combat concludes and viewers can stop watching.

Return a timestamp between {clip_end_time:.0f} and {analysis_start + buffer_seconds:.0f}.

Look for: last enemy eliminated, victory screen, combat music fades, score finalizes, scene transition.

Return only the timestamp in seconds like: {clip_end_time + 5.0:.0f}"""
        
        try:
            # Use LM Studio SDK if available
            if self.lm_client:
                # SDK path: limit to 6 frames for single-call approach
                sdk_frames = frames
                if len(sdk_frames) > 6:
                    step = len(sdk_frames) // 6
                    sdk_frames = sdk_frames[::step][:6]
                
                system_msg = {"role": "system", "content": "You are a gaming video analyst. Analyze frames and return only timestamp numbers in seconds."}
                user_content = [{"type": "text", "text": prompt}]
                
                for timestamp, frame in sdk_frames:
                    import base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    user_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                    })
                
                user_msg = {"role": "user", "content": user_content}
                
                try:
                    model = self.lm_client.llm.model(self.vlm_model)
                    response = model.respond(
                        history={"messages": [system_msg, user_msg]},
                        config={"temperature": 0.1, "max_tokens": 50}
                    )
                    result = response.content.strip()
                except Exception as sdk_error:
                    print(f"   SDK method failed: {sdk_error}")
                    print("   Using HTTP batched fallback")
                    self.lm_client = None
                    # Pass ALL frames to batched HTTP fallback
                    return self._analyze_with_http_batched(frames, prompt, analysis_type,
                                                          analysis_start, clip_start_time, clip_end_time, buffer_seconds)
                
            else:
                # HTTP batched fallback with ALL frames
                print("   Using HTTP batched fallback")
                return self._analyze_with_http_batched(frames, prompt, analysis_type,
                                                      analysis_start, clip_start_time, clip_end_time, buffer_seconds)
            
            # Parse SDK response
            import re
            timestamp_match = re.search(r'(\d+\.?\d*)', result)
            if timestamp_match:
                return {
                    "safe_timestamp": float(timestamp_match.group(1)),
                    "confidence": 0.8,
                    "reason": f"VLM analysis: {analysis_type} point"
                }
            else:
                raise ValueError(f"No timestamp found in: {result[:100]}")
                
        except Exception as e:
            print(f"   VLM analysis failed: {e}")
            if analysis_type == "entry":
                safe_timestamp = frames[0][0] if frames else None
            else:
                safe_timestamp = frames[-1][0] if frames else None
            return {
                "safe_timestamp": safe_timestamp,
                "confidence": 0.6,
                "reason": "Fallback: " + ("earliest" if analysis_type == "entry" else "latest") + " frame"
            }
    
    def _get_api_url(self):
        """Build the correct LM Studio API URL"""
        base_url = self.vlm_url
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        if '/v1' not in base_url:
            return f"{base_url}/v1/chat/completions"
        return f"{base_url}/chat/completions"
    
    def _prepare_frame_b64(self, frame: np.ndarray) -> str:
        """Resize and encode a frame as base64 JPEG"""
        import base64
        h, w = frame.shape[:2]
        scale = min(512 / w, 512 / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _send_vlm_request(self, content, timeout=120):
        """Send a single request to LM Studio and return the text response"""
        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": "You are a gaming video analyst. Analyze frames and return only timestamp numbers in seconds."},
                {"role": "user", "content": content}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        url = self._get_api_url()
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()
        if 'choices' not in response_json or len(response_json['choices']) == 0:
            raise ValueError(f"Unexpected API response: {list(response_json.keys())}")
        return response_json['choices'][0]['message']['content'].strip()

    def _analyze_with_http_batched(self, frames: List[Tuple[float, np.ndarray]], prompt: str, 
                                    analysis_type: str, analysis_start: float = 0,
                                    clip_start_time: float = 0, clip_end_time: float = 0,
                                    buffer_seconds: float = 60) -> Dict:
        """
        Batched HTTP VLM analysis with rolling context.
        
        - Sends frames in batches of 12
        - Each batch receives a rolling summary of what previous batches found
        - Final aggregation picks the best timestamp from all candidates
        """
        import re
        
        batch_size = 12
        batch_results = []
        rolling_summary = ""  # Accumulates context across batches
        total_batches = (len(frames) + batch_size - 1) // batch_size
        
        search_target = "FIGHT START" if analysis_type == "entry" else "FIGHT END"
        
        print(f"   📦 Batched analysis: {len(frames)} frames in {total_batches} batches of {batch_size} (with rolling context)")
        
        # ---- Pass 1: Send each batch with rolling context ----
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            
            t_start = batch_frames[0][0]
            t_end = batch_frames[-1][0]
            
            # Build prompt with rolling context from previous batches
            context_section = ""
            if rolling_summary:
                context_section = f"""
PREVIOUS BATCHES SUMMARY (what was seen so far):
{rolling_summary}
Use this context to understand the flow of action. Build on what was observed before.
"""
            
            batch_prompt = f"""Batch {batch_idx + 1}/{total_batches}: Analyze these {len(batch_frames)} gaming frames from {t_start:.1f}s to {t_end:.1f}s.
{context_section}
You are looking for the {search_target} point.
{"Look for: first shot fired, weapon drawn, enemy appears, health/shield changes, combat-ready movement." if analysis_type == "entry" else "Look for: last enemy eliminated, victory screen, combat music fades, player relaxes, scene transition to non-combat."}

Briefly describe what's happening across these frames (combat? looting? running? lobby?).
Then decide if this batch contains a good {analysis_type} point.

Format your answer as:
DESCRIPTION: [what's happening in these frames]
CANDIDATE: [timestamp in seconds, or NONE if no good {analysis_type} point in this batch]"""
            
            content = [{"type": "text", "text": batch_prompt}]
            for timestamp, frame in batch_frames:
                frame_b64 = self._prepare_frame_b64(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                })
            
            try:
                print(f"   Batch {batch_idx + 1}/{total_batches} ({t_start:.0f}s-{t_end:.0f}s)...", end=" ")
                result = self._send_vlm_request(content, timeout=120)
                
                # Extract description
                desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=CANDIDATE:|$)', result, re.IGNORECASE | re.DOTALL)
                description = desc_match.group(1).strip()[:150] if desc_match else result[:150]
                
                # Extract candidate timestamp
                candidate_match = re.search(r'CANDIDATE:\s*(\d+\.?\d*)', result, re.IGNORECASE)
                timestamp_val = None
                if candidate_match:
                    timestamp_val = float(candidate_match.group(1))
                elif 'NONE' not in result.upper():
                    # Try to find any number as fallback
                    num_match = re.search(r'(\d+\.?\d*)', result)
                    if num_match:
                        timestamp_val = float(num_match.group(1))
                
                batch_results.append({
                    "batch": batch_idx + 1,
                    "range": f"{t_start:.1f}s-{t_end:.1f}s",
                    "candidate": timestamp_val,
                    "description": description
                })
                
                # Update rolling summary for next batch
                candidate_str = f"candidate={timestamp_val:.1f}s" if timestamp_val else "no candidate"
                rolling_summary += f"- Batch {batch_idx + 1} ({t_start:.0f}s-{t_end:.0f}s): {description[:80]} [{candidate_str}]\n"
                
                status = f"candidate={timestamp_val}" if timestamp_val else "no candidate"
                print(f"✅ {status}")
                
            except Exception as e:
                print(f"❌ {e}")
                batch_results.append({
                    "batch": batch_idx + 1,
                    "range": f"{t_start:.1f}s-{t_end:.1f}s",
                    "candidate": None,
                    "description": f"Error: {str(e)[:100]}"
                })
                rolling_summary += f"- Batch {batch_idx + 1} ({t_start:.0f}s-{t_end:.0f}s): [analysis failed]\n"
        
        # ---- Pass 2: Aggregate results ----
        candidates = [b for b in batch_results if b["candidate"] is not None]
        
        if not candidates:
            print(f"   ⚠️  No candidates found from any batch, using fallback")
            if analysis_type == "entry":
                return {"safe_timestamp": frames[0][0], "confidence": 0.5, "reason": "Fallback: no VLM candidates"}
            else:
                return {"safe_timestamp": frames[-1][0], "confidence": 0.5, "reason": "Fallback: no VLM candidates"}
        
        if len(candidates) == 1:
            print(f"   🎯 Single candidate: {candidates[0]['candidate']}s")
            return {
                "safe_timestamp": candidates[0]["candidate"],
                "confidence": 0.75,
                "reason": f"VLM batched analysis: {analysis_type} (1 candidate)"
            }
        
        # Multiple candidates — final aggregation call (text-only)
        print(f"   🔄 Aggregating {len(candidates)} candidates...")
        
        if analysis_type == "entry":
            valid_range = f"{analysis_start:.0f}s to {clip_start_time:.0f}s"
        else:
            valid_range = f"{clip_end_time:.0f}s to {clip_end_time + buffer_seconds:.0f}s"
        
        agg_prompt = f"""You analyzed {total_batches} batches of gaming footage with rolling context, looking for the {search_target} point.

Valid timestamp range: {valid_range}

Full rolling analysis:
{rolling_summary}

Based on this full chronological analysis, which candidate timestamp is the BEST {analysis_type} point?
Consider the flow of action across all batches — pick the moment where {"combat clearly begins" if analysis_type == "entry" else "combat clearly ends"}.

Return only the timestamp in seconds like: {candidates[0]['candidate']:.0f}"""
        
        try:
            agg_content = [{"type": "text", "text": agg_prompt}]
            result = self._send_vlm_request(agg_content, timeout=30)
            
            timestamp_match = re.search(r'(\d+\.?\d*)', result)
            if timestamp_match:
                final_ts = float(timestamp_match.group(1))
                print(f"   🎯 Aggregated result: {final_ts}s")
                return {
                    "safe_timestamp": final_ts,
                    "confidence": 0.85,
                    "reason": f"VLM batched analysis: {analysis_type} ({len(candidates)} candidates, rolling context)"
                }
        except Exception as e:
            print(f"   Aggregation failed: {e}")
        
        # Fallback: heuristic pick
        if analysis_type == "entry":
            best = max(candidates, key=lambda b: b["candidate"])
        else:
            best = min(candidates, key=lambda b: b["candidate"])
        
        print(f"   🎯 Heuristic pick: {best['candidate']}s (from batch {best['batch']})")
        return {
            "safe_timestamp": best["candidate"],
            "confidence": 0.7,
            "reason": f"VLM batched analysis: {analysis_type} (heuristic from {len(candidates)} candidates)"
        }
    
    def find_safe_entry(self, clip_start_time: float, buffer_minutes: float = 1.2, clip_events: List[Dict] = None) -> Dict:
        """
        Find safe entry point before clip start.
        
        Two-tier approach:
          1. Audio analysis: fast RMS energy onset detection (no GPU for VLM).
          2. VLM batched analysis (fallback if audio confidence < 0.65).
        """
        max_buffer = 60.0
        buffer_seconds = min(buffer_minutes * 60, max_buffer)
        analysis_start = max(0, clip_start_time - buffer_seconds)
        min_entry_time = clip_start_time - max_buffer

        print(f"🔍 Finding safe entry: {analysis_start:.1f}s to {clip_start_time:.1f}s (max {max_buffer}s buffer)")

        # ── Tier 1: Audio analysis ─────────────────────────────────────────────
        if self.audio_analyzer is not None:
            try:
                audio_ts, audio_conf = self.audio_analyzer.find_audio_entry_point(
                    analysis_start, clip_start_time, clip_start=clip_start_time
                )
                print(f"   [Audio] Audio entry: {audio_ts}s (confidence={audio_conf:.2f})")

                if audio_ts is not None and audio_conf >= 0.65:
                    audio_ts = max(audio_ts - 4.0, min_entry_time) # 4s pacing pad
                    print(f"   ✅ Audio entry (with 4s pad): {audio_ts:.1f}s")
                elif audio_ts is not None:
                    # Low confidence — use audio timestamp but note it
                    audio_ts = max(audio_ts - 4.0, min_entry_time) # 4s pacing pad
                    print(f"   [Audio] Low-confidence audio entry (with 4s pad): {audio_ts:.1f}s (conf={audio_conf:.2f})")
                else:
                    # No onset found — default to 10s before clip
                    audio_ts = max(clip_start_time - 10.0, min_entry_time)
                    audio_conf = 0.4
                    print(f"   [Audio] No audio onset found, using default: {audio_ts:.1f}s")
            except Exception as e:
                print(f"   [Audio] Audio error: {e}, using default")
                audio_ts = max(clip_start_time - 10.0, min_entry_time)
                audio_conf = 0.3

            # Audio mode: never call VLM
            print(f"   Safe entry: {audio_ts:.1f}s (confidence: {audio_conf:.2f})")
            return {
                "safe_timestamp": audio_ts,
                "confidence": audio_conf,
                "reason": f"Audio analysis: entry at {audio_ts:.1f}s"
            }

        # ── VLM batched analysis (only when no audio_analyzer) ─────────────────
        frames = self._extract_frames(analysis_start, buffer_seconds)
        print(f"   Extracted {len(frames)} frames at {self.fps} fps")

        event_context = ""
        if clip_events:
            event_types = [event.get('text', '') or '' for event in clip_events[:3]]
            event_types = [t for t in event_types if t.strip()]
            event_context = f" UPCOMING EVENTS: {', '.join(event_types)}" if event_types else ""

        results = self._analyze_frames_with_vlm(
            frames, "entry", event_context,
            analysis_start, clip_start_time, clip_start_time, buffer_seconds
        )

        # Clamp to max buffer window
        if results['safe_timestamp'] is not None:
            if results['safe_timestamp'] < min_entry_time:
                results['safe_timestamp'] = min_entry_time
                results['confidence'] = min(results['confidence'], 0.6)
                results['reason'] = f"Adjusted for 30s max buffer: {results['reason']}"
        else:
            results['safe_timestamp'] = max(clip_start_time - 10.0, min_entry_time)
            results['confidence'] = 0.5
            results['reason'] = "VLM failed: using 10s before clip"

        print(f"   Safe entry: {results['safe_timestamp']:.1f}s (confidence: {results['confidence']:.2f})")
        return results
    
    def find_safe_exit(self, clip_end_time: float, buffer_minutes: float = 1.5, clip_events: List[Dict] = None) -> Dict:
        """
        Find safe exit point after clip end.
        
        Two-tier approach:
          1. Audio analysis: fast RMS energy drop-off detection.
          2. VLM batched analysis (fallback if audio confidence < 0.65).
        """
        max_buffer = 45.0
        buffer_seconds = min(buffer_minutes * 60, max_buffer)
        analysis_start = clip_end_time
        max_exit_time = clip_end_time + max_buffer

        print(f"🔍 Finding safe exit: {clip_end_time:.1f}s to {analysis_start + buffer_seconds:.1f}s (max {max_buffer}s buffer)")

        # ── Tier 1: Audio analysis ─────────────────────────────────────────────
        if self.audio_analyzer is not None:
            try:
                audio_ts, audio_conf = self.audio_analyzer.find_audio_exit_point(
                    clip_end_time, clip_end_time + buffer_seconds, clip_end=clip_end_time
                )
                print(f"   [Audio] Audio exit: {audio_ts}s (confidence={audio_conf:.2f})")

                if audio_ts is not None and audio_conf >= 0.65:
                    audio_ts = min(audio_ts, max_exit_time)
                    print(f"   ✅ Audio exit: {audio_ts:.1f}s")
                elif audio_ts is not None:
                    audio_ts = min(audio_ts, max_exit_time)
                    print(f"   [Audio] Low-confidence audio exit: {audio_ts:.1f}s (conf={audio_conf:.2f})")
                else:
                    # No drop-off found — default to 10s after clip
                    audio_ts = min(clip_end_time + 10.0, max_exit_time)
                    audio_conf = 0.4
                    print(f"   [Audio] No audio drop-off found, using default: {audio_ts:.1f}s")
            except Exception as e:
                print(f"   [Audio] Audio error: {e}, using default")
                audio_ts = min(clip_end_time + 10.0, max_exit_time)
                audio_conf = 0.3

            # Audio mode: never call VLM
            print(f"   Safe exit: {audio_ts:.1f}s (confidence: {audio_conf:.2f})")
            return {
                "safe_timestamp": audio_ts,
                "confidence": audio_conf,
                "reason": f"Audio analysis: exit at {audio_ts:.1f}s"
            }

        # ── VLM batched analysis (only when no audio_analyzer) ─────────────────
        frames = self._extract_frames(analysis_start, buffer_seconds)
        print(f"   Extracted {len(frames)} frames at {self.fps} fps")

        event_context = ""
        if clip_events:
            event_types = [event.get('text', '') or '' for event in clip_events[-3:]]
            event_types = [t for t in event_types if t.strip()]
            event_context = f" COMPLETED EVENTS: {', '.join(event_types)}" if event_types else ""

        results = self._analyze_frames_with_vlm(
            frames, "exit", event_context,
            analysis_start, 0, clip_end_time, buffer_seconds
        )

        # Clamp to max buffer window
        if results['safe_timestamp'] is not None:
            if results['safe_timestamp'] > max_exit_time:
                results['safe_timestamp'] = max_exit_time
                results['confidence'] = min(results['confidence'], 0.6)
                results['reason'] = f"Adjusted for 30s max buffer: {results['reason']}"
        else:
            results['safe_timestamp'] = min(clip_end_time + 10.0, max_exit_time)
            results['confidence'] = 0.5
            results['reason'] = "VLM failed: using 10s after clip"

        print(f"   Safe exit: {results['safe_timestamp']:.1f}s (confidence: {results['confidence']:.2f})")
        return results
    
    def process_clips(self, clips_json: str, entry_buffer: float = 1.2, exit_buffer: float = 1.5):
        """
        Process all clips to find safe entry/exit points and update the clips.json file.
        
        Args:
            clips_json: Clips JSON file to update in-place
            entry_buffer: Minutes before clip for entry analysis
            exit_buffer: Minutes after clip for exit analysis
        """
        # Load clips
        with open(clips_json, 'r', encoding='utf-8') as f:
            clips = json.load(f)
        
        print(f"🎬 Processing {len(clips)} clips for safe entry/exit...")
        
        # Process each clip
        processed_clips = []
        for i, clip in enumerate(clips):
            print(f"\n--- Clip {i+1}/{len(clips)} ---")
            print(f"   Original: {clip['start_time']:.1f}s - {clip['end_time']:.1f}s")
            
            # Find safe entry with event context
            entry_results = self.find_safe_entry(
                clip['start_time'], 
                entry_buffer, 
                clip.get('events', [])
            )
            
            # Find safe exit with event context  
            exit_results = self.find_safe_exit(
                clip['end_time'], 
                exit_buffer,
                clip.get('events', [])
            )
            
            # Check for overlap with previous processed clips
            safe_start = entry_results['safe_timestamp']
            safe_end = exit_results['safe_timestamp']
            
            overlap_found = False
            for prev_clip in processed_clips:
                prev_start = prev_clip['safe_entry']['timestamp']
                prev_end = prev_clip['safe_exit']['timestamp']
                
                # Check if current clip overlaps with previous clip
                if not (safe_end <= prev_start or safe_start >= prev_end):
                    print(f"   ⚠️  Overlap detected with previous clip!")
                    print(f"      Previous: {prev_start:.1f}s - {prev_end:.1f}s")
                    print(f"      Current:  {safe_start:.1f}s - {safe_end:.1f}s")
                    
                    # Adjust current clip to avoid overlap
                    if safe_start < prev_end:
                        # Move start to after previous clip ends
                        safe_start = prev_end + 1.0  # Add 1s buffer
                        entry_results['safe_timestamp'] = safe_start
                        entry_results['confidence'] = 0.3  # Lower confidence due to adjustment
                        entry_results['reason'] = f"Adjusted to avoid overlap: {entry_results['reason']}"
                        print(f"   ✅ Adjusted start to {safe_start:.1f}s (after previous clip)")
                    
                    overlap_found = True
                    break
            
            if not overlap_found:
                print(f"   ✅ No overlap detected")
            
            # Add safe timestamps to clip
            clip['safe_entry'] = {
                'timestamp': entry_results['safe_timestamp'],
                'confidence': entry_results['confidence'],
                'reason': entry_results['reason']
            }
            
            clip['safe_exit'] = {
                'timestamp': exit_results['safe_timestamp'],
                'confidence': exit_results['confidence'],
                'reason': exit_results['reason']
            }
            
            clip['safe_duration'] = exit_results['safe_timestamp'] - entry_results['safe_timestamp']
            
            print(f"   Safe: {entry_results['safe_timestamp']:.1f}s - {exit_results['safe_timestamp']:.1f}s")
            print(f"   Duration: {clip['safe_duration']:.1f}s")
            
            processed_clips.append(clip)
        
        # Save results back to the same file
        with open(clips_json, 'w', encoding='utf-8') as f:
            json.dump(clips, f, indent=2)
        
        print(f"\n✅ Safe entry/exit analysis complete!")
        print(f"   Processed {len(clips)} clips")
        print(f"   Updated: {clips_json}")
        
        return clips
    
    def close(self):
        """Close video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None


def main():
    """Main function for safe entry/exit detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Entry/Exit Detection with VLM")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--clips", required=True, help="Clips JSON file to update in-place")
    parser.add_argument("--output", help="Output JSON file (optional - updates clips.json if not specified)")
    parser.add_argument("--vlm_url", default="http://localhost:1234", help="VLM server URL")
    parser.add_argument("--vlm_model", default="google/gemma-3-27b", help="VLM model name")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for VLM analysis")
    parser.add_argument("--entry_buffer", type=float, default=1.2, help="Minutes before clip for entry analysis")
    parser.add_argument("--exit_buffer", type=float, default=1.5, help="Minutes after clip for exit analysis")
    
    args = parser.parse_args()
    
    # Create detector
    detector = SafeEntryExitDetector(
        video_path=args.video,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        fps=args.fps
    )
    
    try:
        # Process clips (update in-place or save to new file)
        if args.output:
            # Save to new file
            clips = detector.process_clips(
                clips_json=args.clips,
                output_json=args.output,
                entry_buffer=args.entry_buffer,
                exit_buffer=args.exit_buffer
            )
            output_file = args.output
        else:
            # Update clips.json in-place
            clips = detector.process_clips(
                clips_json=args.clips,
                entry_buffer=args.entry_buffer,
                exit_buffer=args.exit_buffer
            )
            output_file = args.clips
        
        print(f"\n🎉 Safe Entry/Exit Complete!")
        print(f"   Video: {args.video}")
        print(f"   Input clips: {args.clips}")
        print(f"   Updated: {output_file}")
        print(f"   VLM FPS: {args.fps}")
        
    finally:
        detector.close()


if __name__ == "__main__":
    main()
