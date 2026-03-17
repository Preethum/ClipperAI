"""
DetectionM.py - GameEventDetector: YOLO + OCR event detection only.
Stripped down to essential event detection functionality.
"""

import cv2
import json
import os
import argparse
import re
import difflib
from datetime import timedelta
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: 'ultralytics' module not found.")
    YOLO = None

try:
    import easyocr
except ImportError:
    print("Warning: 'easyocr' module not found.")
    easyocr = None

def validate_ocr_text(text, label, weights=None):
    """
    Validate that OCR text contains expected keywords for the detected label.
    Supports weighted scoring for different keyword importance.
    Uses fuzzy matching to autocorrect slight OCR misspellings (e.g., 'elininated').
    """
    if not text or text.strip() == "":
        return False
    
    # Default weights if not provided
    default_weights = {
        "Victory": {"you are the champion": 3.0, "champion": 2.0, "victory": 2.0, "win": 1.0},
        "ELIMINATED Text": {"squad": 3.0, "eliminated": 2.0, "wipe": 2.5, "match summary": 1.0, "game over": 1.0},
        "Combat Feed / Kill Feed Updates": {"you are the champion": 5.0, "squad eliminated": 4.0, "champion eliminated": 3.0, "eliminated": 2.0, "knocked down": 2.0, "assist": 1.5},
        "Floating Damage Numbers": {"+": 1.0, "1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5, "5": 0.5, "6": 0.5, "7": 0.5, "8": 0.5, "9": 0.5, "0": 0.5}
    }
    
    label_weights = weights or default_weights.get(label, {})
    text_lower = text.lower()
    total_score = 0.0
    
    # Special handling for damage numbers: match ANY number
    if label in ["Floating Damage Numbers", "loating Damage Numbers"]:
        if re.search(r'\d+', text):
            return True
        return False
    
    expected_keywords = label_weights.keys()
    
    for keyword in expected_keywords:
        if keyword in text_lower:
            total_score += label_weights.get(keyword, 1.0)
            continue
            
        text_words = text_lower.split()
        keyword_words = keyword.split()
        
        if len(keyword_words) == 1:
            for word in text_words:
                similarity = difflib.SequenceMatcher(None, keyword, word).ratio()
                if similarity > 0.8:
                    total_score += label_weights.get(keyword, 1.0) * similarity
                    break
        else:
            similarity = difflib.SequenceMatcher(None, keyword, text_lower).ratio()
            if similarity > 0.6: 
                total_score += label_weights.get(keyword, 1.0) * similarity
    
    return total_score > 0

def correct_ocr_text(text, llm_url=None, model=None):
    """Simple text cleanup removing LLM system messages."""
    if not text or text.strip() == "":
        return text
    
    if '\n' in text:
        lines = text.split('\n')
        for line in lines:
            if line.strip() and not any(word in line.lower() for word in ['okay', 'need', 'figure', 'fix', 'text', 'user', 'mentioned', 'so', 'alright', 'step']):
                text = line.strip()
                break
        else:
            return ""
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class GameEventDetector:
    """Core event detection using YOLO + OCR"""
    
    def __init__(self, model_path: str, enable_ocr: bool = True):
        if YOLO is None:
            raise ImportError("ultralytics package required for YOLO detection")
            
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.enable_ocr = enable_ocr
        
        if enable_ocr and easyocr:
            self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        else:
            self.ocr_reader = None
    
    def detect_events(self, video_path: str, interval_seconds: float = 1.0,
                      min_confidence: float = 0.5, target_labels: List[str] = None,
                      label_weights: Dict = None, debug: bool = False,
                      include_windows: List[Tuple[float, float]] = None) -> List[Dict]:
        """
        Scan video and detect gameplay events.
        
        Args:
            video_path: Path to video file
            interval_seconds: Seconds between frame scans
            min_confidence: Minimum confidence threshold for events
            target_labels: List of labels to keep (None keeps all)
            label_weights: Custom weights for OCR validation
            debug: Print debug information
            include_windows: List of (start_s, end_s) windows to scan. If None, scans entire video.
            
        Returns:
            List of detected events with timestamps
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = int(fps * interval_seconds)
        
        print(f"Scanning video: {os.path.basename(video_path)}")
        print(f"FPS: {fps:.1f}, Frames: {total_frames}, Interval: {interval_seconds}s")
        if include_windows:
            total_scan_s = sum(w[1] - w[0] for w in include_windows)
            print(f"Pruning active: Scanning {len(include_windows)} windows (~{total_scan_s:.1f}s total)")
        
        events = []
        
        # Batching variables
        # Adjust batch size for TensorRT engines which are often compiled with static batch=1
        batch_size = 16 if str(self.model_path).lower().endswith(".engine") else 128
        batch_frames = []
        batch_timestamps = []
        
        # Calculate target frames if using windows
        target_frames = []
        if include_windows:
            for start_s, end_s in include_windows:
                start_f = int(start_s * fps)
                end_f = min(int(end_s * fps), total_frames)
                # align start_f to stride
                current_f = (start_f // stride) * stride
                while current_f < end_f:
                    if current_f < total_frames:
                        target_frames.append(current_f)
                    current_f += stride
            # Remove duplicates and sort
            target_frames = sorted(list(set(target_frames)))
        else:
            target_frames = list(range(0, total_frames, stride))

        if not target_frames:
            cap.release()
            return []

        # Create progress bar
        with tqdm(total=len(target_frames), desc="Processing frames", unit="frames") as pbar:
            current_idx = 0
            current_pos = 0  # MANUALLY TRACKED FRAME POSITION
            
            while current_idx < len(target_frames):
                frame_idx = target_frames[current_idx]
                
                if frame_idx != current_pos:
                    # If gap is small (< 5 frames), grabbing is usually faster than seeks
                    if 0 < (frame_idx - current_pos) < 5:
                        for _ in range(frame_idx - current_pos):
                            if cap.grab():
                                current_pos += 1
                            else:
                                break
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        current_pos = frame_idx  # ASSUME SEEK SUCCEEDED
                
                ret, frame = cap.read()
                if not ret:
                    break
                current_pos += 1  # Increment after read
                if not ret:
                    break
                
                # Downscale for performance
                height, width = frame.shape[:2]
                if height > 1080:
                    scale = 1080 / height
                    frame = cv2.resize(frame, (int(width * scale), 1080))
                    
                timestamp = frame_idx / fps
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
                
                if len(batch_frames) >= batch_size:
                    events.extend(self._run_batch_inference(
                        batch_frames, batch_timestamps, min_confidence, target_labels, label_weights, debug
                    ))
                    batch_frames = []
                    batch_timestamps = []
                
                current_idx += 1
                pbar.update(1)
                
            # Final partial batch
            if batch_frames:
                events.extend(self._run_batch_inference(
                    batch_frames, batch_timestamps, min_confidence, target_labels, label_weights, debug
                ))
        
        cap.release()
        print(f"Detected {len(events)} events")
        return events

    def _run_batch_inference(self, frames: List[np.ndarray], timestamps: List[float],
                             min_confidence: float, target_labels: List[str],
                             label_weights: Dict, debug: bool) -> List[Dict]:
        """Runs YOLO on a batch of frames, then Batches OCR on all resulting detections."""
        import torch
        
        batch_events = []
        
        # 1. Batched YOLO
        with torch.no_grad():
            results = self.model.predict(frames, conf=min_confidence, verbose=False, half=False, stream=False, batch=len(frames))
        
        # 2. Collect all detections that need OCR
        pending_ocr = [] # List of (crop, event_dict)
        
        for i, result in enumerate(results):
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = result.names[cls_id]
                conf = float(box.conf[0].item())
                
                if target_labels and label not in target_labels:
                    continue
                if conf < min_confidence:
                    continue
                    
                event = {
                    "timestamp": self._format_timestamp(timestamps[i]),
                    "timestamp_s": round(timestamps[i], 2),
                    "label": label,
                    "confidence": round(conf, 3),
                    "text": None
                }
                
                # Extract crop for OCR
                if self.enable_ocr and self.ocr_reader:
                    labels_exempt_from_text = ["Directional Damage Indicators", "loating Damage Numbers"]
                    if label not in labels_exempt_from_text:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            crop = frames[i][y1:y2, x1:x2]
                            if crop.size > 0:
                                pending_ocr.append((crop, event))
                        except Exception as e:
                            if debug: print(f"Crop failed: {e}")
                    else:
                        event["text"] = "[Visual Indicator]"
                        batch_events.append(event)
                else:
                    batch_events.append(event)

        # 3. Batch process OCR
        # EasyOCR readtext is still somewhat sequential, but we can avoid overhead by using the list
        for crop, event in pending_ocr:
            try:
                ocr_results = self.ocr_reader.readtext(crop)
                text_blocks = [res[1] for res in ocr_results if res[2] > 0.5]
                if text_blocks:
                    text = " ".join(text_blocks)
                    text = correct_ocr_text(text)
                    
                    if validate_ocr_text(text, event["label"], label_weights) or event["confidence"] >= 0.85:
                        event["text"] = text
                        batch_events.append(event)
                elif event["confidence"] >= 0.85:
                    # High confidence YOLO bypasses OCR keywords
                    event["text"] = "[Unreadable text]"
                    batch_events.append(event)
            except Exception as e:
                if debug: print(f"OCR error: {e}")

        return batch_events
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        td = timedelta(seconds=seconds)
        time_str = str(td).split('.')[0]
        parts = time_str.split(':')
        if len(parts) == 2:
            return f"00:{int(parts[0]):02d}:{int(parts[1]):02d}"
        elif len(parts) == 3:
            return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
        return time_str


def main():
    """Command line interface for event detection only"""
    parser = argparse.ArgumentParser(description="Detect gameplay events from video")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("model", help="YOLO model path (.pt)")
    parser.add_argument("--output", default="events.json", help="Output JSON file")
    parser.add_argument("--interval", type=float, default=1.0, help="Scanning interval (seconds)")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR processing")
    
    args = parser.parse_args()
    
    detector = GameEventDetector(args.model, enable_ocr=not args.no_ocr)
    events = detector.detect_events(args.video, args.interval)
    
    # Save events to JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"Events saved to: {args.output}")
    print(f"Total events detected: {len(events)}")


if __name__ == "__main__":
    main()
