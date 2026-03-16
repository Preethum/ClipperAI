"""
Clipper_gamingM.py - Dynamic Event Clumping with Sliding Window Analysis
Groups nearby events into cohesive clips using sliding window approach.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


class EventClumper:
    """Groups gaming events into clips using sliding window analysis"""
    
    def __init__(self, events_json_path: str, window_seconds: int = 60, intensity_weights: Dict = None):
        self.events_json_path = events_json_path
        self.window_seconds = window_seconds
        self.events = []
        self.clips = []
        self.clip_counter = 0  # Add counter for unique clip IDs
        
        # Use provided weights or defaults
        self.intensity_weights = intensity_weights or {
            'base_score': 1.0,
            'combat_bonus': 2.0,
            'high_confidence_bonus': 0.5,
            'squad_wipe_bonus': 1.5,
            'high_damage_bonus': 1.0,
            'elimination_bonus': 1.0,
            # Label-specific bonuses for Apex Legends
            'label_bonuses': {
                'Victory': 5.0,
                'ELIMINATED Text': 4.0,
                'loating Damage Numbers': 2.0,
                'Directional Damage Indicators': 1.5
            }
        }
        
        print(f"Intensity weights: {self.intensity_weights}")
        
        # Load events
        self._load_events()
        
    def _load_events(self):
        """Load events from JSON file"""
        try:
            with open(self.events_json_path, 'r', encoding='utf-8') as f:
                self.events = json.load(f)
            print(f"Loaded {len(self.events)} events from {self.events_json_path}")
        except Exception as e:
            print(f"Error loading events: {e}")
            self.events = []
    
    def _is_combat_event(self, event: Dict) -> bool:
        """Check if event represents combat/action"""
        combat_keywords = [
            "kill", "eliminated", "knocked", "down", "assist", 
            "damage", "+100", "+50", "+150", "+75", "+25",
            "squad", "wipe", "elimination"
        ]
        
        # Check label for combat relevance
        combat_labels = ["loating Damage Numbers", "audio_combat_spike"]
        if event.get('label') in combat_labels:
            return True
            
        text = event.get('text', '') or ''
        text = text.lower()
        return any(keyword in text for keyword in combat_keywords)
    
    def _calculate_intensity_score(self, events_in_window: List[Dict]) -> float:
        """Calculate intensity score for events in window using configurable weights"""
        if not events_in_window:
            return 0.0
        
        score = 0.0
        for event in events_in_window:
            label = event.get('label', '')
            if label in ['Combat Feed / Kill Feed Updates', 'POV Player', 'Teammate']:
                continue
                
            # Base score for each event
            base_score = self.intensity_weights['base_score']
            
            # Bonus for label type
            label = event.get('label', '')
            label_bonuses = self.intensity_weights.get('label_bonuses', {})
            base_score += label_bonuses.get(label, 0.0)
            
            # Bonus for combat events
            if self._is_combat_event(event):
                base_score += self.intensity_weights['combat_bonus']
            
            # Explicit high score for audio bursts to ensure they bridge clips
            if label == "audio_combat_spike":
                base_score += 2.0  # Guarantees audio spikes contribute heavily

            # Granular bonus for damage numbers
            text = event.get('text', '') or ''
            if label == "loating Damage Numbers" or "+" in text:
                import re
                damage_match = re.search(r'\+(\d+)', text)
                if damage_match:
                    damage_val = int(damage_match.group(1))
                    # Add 10% of damage value as bonus score
                    base_score += (damage_val * 0.1)
            
            # Bonus for high confidence
            if event.get('confidence', 0) > 0.8:
                base_score += self.intensity_weights['high_confidence_bonus']
            
            # Bonus for specific high-value events
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in ["squad", "wipe"]):
                base_score += self.intensity_weights['squad_wipe_bonus']
            if any(keyword in text_lower for keyword in ["eliminated", "elimination"]):
                base_score += self.intensity_weights['elimination_bonus']
                
            score += base_score
        
        return score
    
    def _find_story_arc_boundaries(self, events_in_window: List[Dict]) -> Tuple[float, float]:
        """Calculate story arc boundaries (buildup and celebration)"""
        if not events_in_window:
            return 0.0, 0.0
        
        # Find first significant event (buildup start)
        first_event = min(events_in_window, key=lambda x: x.get('timestamp_s', 0))
        buildup_start = first_event.get('timestamp_s', 0.0)
        
        # Find last significant event (celebration end)
        last_event = max(events_in_window, key=lambda x: x.get('timestamp_s', 0))
        celebration_end = last_event.get('timestamp_s', 0.0) + 10.0  # 10s celebration buffer
        
        return buildup_start, celebration_end
    
    def _create_clip_metadata(self, events_in_window: List[Dict], start_time: float, end_time: float) -> Dict:
        """Create metadata for a clip"""
        # Get unique labels in this window
        labels = list(set(event.get('label', '') for event in events_in_window))
        
        # Count event types
        combat_events = sum(1 for e in events_in_window if self._is_combat_event(e))
        total_events = len(events_in_window)
        
        # Calculate story arc
        buildup_start, celebration_end = self._find_story_arc_boundaries(events_in_window)
        
        # Create clip metadata and increment counter
        clip_metadata = {
            'clip_id': f"clip_{self.clip_counter}",
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'event_count': total_events,
            'combat_events': combat_events,
            'labels': labels,
            'events': events_in_window,
            'intensity_score': self._calculate_intensity_score(events_in_window),
            'buildup_start': buildup_start,
            'celebration_end': celebration_end,
            'story_arc': {
                'buildup_duration': buildup_start - start_time,
                'action_duration': end_time - buildup_start,
                'celebration_duration': celebration_end - end_time
            }
        }
        
        self.clip_counter += 1  # Increment for next clip
        return clip_metadata
    
    def _merge_nearby_windows(self, windows: List[Dict]) -> List[Dict]:
        """Merge windows that are close together"""
        if not windows:
            return []
        
        merged = []
        current = windows[0].copy()
        
        for window in windows[1:]:
            # If windows are within 30 seconds, merge them
            if window['start_time'] - current['end_time'] <= 30:
                # Extend current window
                current['end_time'] = window['end_time']
                
                # Use unique events (by timestamp and text/label)
                all_events = current['events'] + window['events']
                seen = set()
                unique_events = []
                for e in all_events:
                    # Create a unique key for the event
                    key = (e.get('timestamp_s'), e.get('label'), e.get('text'))
                    if key not in seen:
                        unique_events.append(e)
                        seen.add(key)
                
                current['events'] = unique_events
                current['event_count'] = len(unique_events)
                current['intensity_score'] = self._calculate_intensity_score(unique_events)
                current['labels'] = list(set(current['labels'] + window['labels']))
            else:
                merged.append(current)
                current = window.copy()
        
        merged.append(current)
        
        # Fix clip IDs for merged clips
        for i, clip in enumerate(merged):
            clip['clip_id'] = f"clip_{self.clip_counter + i}"
        
        self.clip_counter += len(merged)  # Update counter
        return merged
    
    def _detect_gaps(self, events: List[Dict], gap_threshold: float = 30.0) -> List[Tuple[int, int]]:
        """
        Detect significant gaps between events that indicate separate clips.
        
        Args:
            events: Sorted list of events
            gap_threshold: Minimum gap in seconds to consider as separate clip
            
        Returns:
            List of (index, gap_size) tuples where gaps were found
        """
        gaps = []
        
        for i in range(1, len(events)):
            prev_event = events[i-1]
            curr_event = events[i]
            
            prev_time = prev_event.get('timestamp_s', 0)
            curr_time = curr_event.get('timestamp_s', 0)
            gap = curr_time - prev_time
            
            # If the gap involves an audio combat spike, be more forgiving (e.g. up to 60s)
            # because audio events might be sparse during sustained combat if the smoothed baseline rises
            is_audio_bridge = (prev_event.get('label') == 'audio_combat_spike' or 
                               curr_event.get('label') == 'audio_combat_spike')
            
            # Use 60.0s for an audio bridge, this binds nearby action together if there's audio.
            effective_threshold = max(gap_threshold, 60.0) if is_audio_bridge else gap_threshold
            
            if gap >= effective_threshold:
                gaps.append((i, gap))
                print(f"   Gap detected: {gap}s between events {i-1} and {i} (threshold: {effective_threshold}s)")
        
        return gaps
    
    def detect_clips(self, min_intensity: float = 3.0, gap_threshold: float = 30.0) -> List[Dict]:
        """
        Detect clips using sliding window approach with gap detection and trigger-only labels.
        
        Args:
            min_intensity: Minimum intensity score to create a clip
            gap_threshold: Minimum gap in seconds to start new clip
            
        Returns:
            List of clip metadata dictionaries
        """
        if not self.events:
            return []
            
        # Define labels that are allowed to trigger/start a new window
        # Background elements (POV Player, Teammate) should NOT start a clip
        trigger_labels = [
            "Victory",
            "ELIMINATED Text",
            "loating Damage Numbers",
            "Directional Damage Indicators",
            "audio_combat_spike"  # Audio events can now trigger and bridge clips
        ]
        
        # Sort events by timestamp
        sorted_events = sorted(self.events, key=lambda x: x.get('timestamp_s', 0))
        
        # Detect gaps first
        gaps = self._detect_gaps(sorted_events, gap_threshold)
        print(f"Found {len(gaps)} significant gaps (>{gap_threshold}s)")
        
        # Split events into segments based on gaps
        segments = []
        start_idx = 0
        
        for gap_idx, gap_size in gaps:
            segment = sorted_events[start_idx:gap_idx]
            if segment:
                segments.append(segment)
                print(f"   Segment {len(segments)}: {len(segment)} events, gap: {gap_size}s")
            start_idx = gap_idx
        
        # Add final segment
        if start_idx < len(sorted_events):
            final_segment = sorted_events[start_idx:]
            segments.append(final_segment)
            print(f"   Final segment: {len(final_segment)} events")
        
        # Process each segment with sliding window
        all_clips = []
        for seg_idx, segment in enumerate(segments):
            print(f"\nProcessing segment {seg_idx + 1}/{len(segments)}...")
            
            # Sliding window analysis on this segment
            windows = []
            window_start = 0
            
            while window_start < len(segment):
                # Optimization: Skip background events that try to start a window
                if segment[window_start].get('label') not in trigger_labels:
                    window_start += 1
                    continue
                
                # Find events within window
                window_end_time = segment[window_start].get('timestamp_s', 0) + self.window_seconds
                
                events_in_window = []
                for i, event in enumerate(segment[window_start:], start=window_start):
                    if event.get('timestamp_s', 0) <= window_end_time:
                        events_in_window.append(event)
                    else:
                        break
                
                if events_in_window:
                    # Calculate window boundaries
                    start_time = events_in_window[0].get('timestamp_s', 0)
                    end_time = events_in_window[-1].get('timestamp_s', 0) + 5.0  # 5s buffer
                    
                    # Create window metadata
                    window_metadata = self._create_clip_metadata(events_in_window, start_time, end_time)
                    
                    # Ensure the window contains BOTH visual combat and audio combat
                    has_visual_combat = False
                    has_audio_combat = False
                    
                    for event in events_in_window:
                        if event.get('label') == 'audio_combat_spike':
                            has_audio_combat = True
                        elif self._is_combat_event(event) and event.get('label') != 'audio_combat_spike':
                            has_visual_combat = True
                            
                        # Fast break if both are found
                        if has_visual_combat and has_audio_combat:
                            break
                    
                    # Only keep windows above intensity threshold AND that have crossing signals
                    if window_metadata['intensity_score'] >= min_intensity:
                        if has_visual_combat and has_audio_combat:
                            windows.append(window_metadata)
                        else:
                            # Skip this window because it lacks either visual combat or audio spikes
                            pass
                
                # Slide window by half the window size or just past the current trigger
                next_jump = max(1, len(events_in_window) // 2)
                window_start += next_jump
            
            # Merge nearby windows in this segment
            segment_clips = self._merge_nearby_windows(windows)
            
            # Add segment info to clips
            for clip in segment_clips:
                clip['segment_id'] = seg_idx + 1
                clip['segment_events'] = len(segment)
                all_clips.append(clip)
        
        self.clips = all_clips
        
        print(f"\nDetected {len(self.clips)} clips from {len(sorted_events)} events")
        print(f"   Segments: {len(segments)}")
        for i, gap in enumerate(gaps):
            print(f"   Gap {i+1}: {gap}s")
        return self.clips
    
    def detect_elimination_clips(self, audio_events: List, 
                                  elim_merge_gap: float = 30.0,
                                  audio_lookback: float = 30.0,
                                  audio_lookahead: float = 30.0,
                                  end_buffer: float = 5.0,
                                  fallback_pad_start: float = 20.0,
                                  fallback_pad_end: float = 10.0,
                                  clip_merge_gap: float = 60.0) -> List[Dict]:
        """
        Create clips anchored to ELIMINATED Text events.
        
        Start = earliest audio_combat_spike before the elimination burst.
        End   = latest audio_combat_spike after the burst + end_buffer.
        Falls back to fixed padding if no audio spike is nearby.
        
        Args:
            audio_events: List of [timestamp_s, label, score] from audio_events.json
            elim_merge_gap: Merge elimination events that are within this many seconds
            audio_lookback: How far before the first elim to search for audio start
            audio_lookahead: How far after the last elim to search for audio end
            end_buffer: Seconds to add after the last found audio spike
            fallback_pad_start: Fixed start padding if no audio spike found
            fallback_pad_end: Fixed end padding if no audio spike found
            clip_merge_gap: Merge final clips whose gap is <= this many seconds (combines back-to-back fights)
        """
        # --- 1. Collect ELIMINATED Text timestamps ---
        elim_events = sorted(
            [e for e in self.events if e.get('label') == 'ELIMINATED Text'],
            key=lambda x: x.get('timestamp_s', 0)
        )
        
        if not elim_events:
            print("No ELIMINATED Text events found. Falling back to detect_clips().")
            return self.detect_clips()
        
        # --- 2. Cluster eliminations into bursts (fights) ---
        bursts = []  # List of (burst_start_s, burst_end_s, [events])
        current_burst_events = [elim_events[0]]
        
        for ev in elim_events[1:]:
            ts = ev.get('timestamp_s', 0)
            last_ts = current_burst_events[-1].get('timestamp_s', 0)
            if ts - last_ts <= elim_merge_gap:
                current_burst_events.append(ev)
            else:
                bursts.append(current_burst_events)
                current_burst_events = [ev]
        bursts.append(current_burst_events)
        
        print(f"Found {len(bursts)} elimination fight(s) from {len(elim_events)} ELIMINATED events")
        
        # --- 3. Build sorted list of audio timestamps for quick lookup ---
        audio_timestamps = sorted([e[0] for e in audio_events]) if audio_events else []
        
        def find_audio_start(elim_start: float) -> float:
            """Find earliest audio spike in [elim_start - lookback, elim_start]."""
            window = [t for t in audio_timestamps 
                      if elim_start - audio_lookback <= t <= elim_start]
            return min(window) if window else elim_start - fallback_pad_start
        
        def find_audio_end(elim_end: float) -> float:
            """Find latest audio spike in [elim_end, elim_end + lookahead]."""
            window = [t for t in audio_timestamps 
                      if elim_end <= t <= elim_end + audio_lookahead]
            return (max(window) + end_buffer) if window else elim_end + fallback_pad_end
        
        # --- 4. Build a clip per burst ---
        raw_clips = []
        for burst_events in bursts:
            elim_start = burst_events[0].get('timestamp_s', 0)
            elim_end = burst_events[-1].get('timestamp_s', 0)
            
            clip_start = max(0, find_audio_start(elim_start))
            clip_end = find_audio_end(elim_end)
            
            # Gather all events in this clip window for metadata
            events_in_clip = [e for e in self.events 
                              if clip_start <= e.get('timestamp_s', 0) <= clip_end]
            
            intensity = self._calculate_intensity_score(events_in_clip)
            
            raw_clips.append({
                'clip_id': f"elim_{int(elim_start)}",
                'start_time': round(clip_start, 2),
                'end_time': round(clip_end, 2),
                'duration': round(clip_end - clip_start, 2),
                'intensity_score': round(intensity, 3),
                'trigger': 'ELIMINATED Text',
                'elim_count': len(burst_events),
                'events': events_in_clip,
                'scores': {
                    'elimination_burst': len(burst_events),
                    'audio_anchored': bool(audio_timestamps)
                }
            })
            
            from datetime import timedelta
            start_fmt = str(timedelta(seconds=int(clip_start)))
            end_fmt = str(timedelta(seconds=int(clip_end)))
            print(f"   Clip [{start_fmt} → {end_fmt}] — {len(burst_events)} elims, intensity: {intensity:.2f}")
        
        # --- 5. Merge overlapping or nearby clips ---
        raw_clips.sort(key=lambda c: c['start_time'])
        merged = []
        for clip in raw_clips:
            if merged and clip['start_time'] <= merged[-1]['end_time'] + clip_merge_gap:
                # Extend the previous clip
                merged[-1]['end_time'] = max(merged[-1]['end_time'], clip['end_time'])
                merged[-1]['duration'] = merged[-1]['end_time'] - merged[-1]['start_time']
                merged[-1]['elim_count'] = merged[-1].get('elim_count', 0) + clip.get('elim_count', 0)
                merged[-1]['events'] = merged[-1].get('events', []) + clip.get('events', [])
                merged[-1]['intensity_score'] = max(merged[-1]['intensity_score'], clip['intensity_score'])
            else:
                merged.append(clip)
        
        self.clips = merged
        print(f"\nDetected {len(self.clips)} audio-anchored elimination clips")
        return self.clips

    def save_clips(self, output_path: str):
        """Save clips to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.clips, f, indent=2)
            print(f"Saved {len(self.clips)} clips to {output_path}")
        except Exception as e:
            print(f"Error saving clips: {e}")


def main():
    """Main function for event clumping"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Event Clumping with Sliding Window")
    parser.add_argument("--events", required=True, help="Events JSON file")
    parser.add_argument("--output", required=True, help="Output clips JSON file")
    parser.add_argument("--window", type=int, default=60, help="Window size in seconds")
    parser.add_argument("--min_intensity", type=float, default=3.0, help="Minimum intensity score")
    parser.add_argument("--gap_threshold", type=float, default=30.0, help="Minimum gap in seconds to start new clip")
    
    # Intensity scoring weights
    parser.add_argument("--intensity_weights", type=str, default=None, 
                    help="Intensity weights as JSON string, e.g., '{\"base_score\": 1.0, \"combat_bonus\": 2.0}'")
    
    args = parser.parse_args()
    
    # Parse intensity weights if provided
    intensity_weights = None
    if args.intensity_weights:
        try:
            intensity_weights = json.loads(args.intensity_weights)
            print(f"Using custom intensity weights: {intensity_weights}")
        except json.JSONDecodeError as e:
            print(f"Error parsing intensity weights: {e}")
            print("Using default weights instead.")
            intensity_weights = None
    
    # Create clumper
    clumper = EventClumper(args.events, args.window, intensity_weights)
    
    # Detect clips with gap threshold
    clips = clumper.detect_clips(args.min_intensity, args.gap_threshold)
    
    # Save results
    clumper.save_clips(args.output)
    
    print(f"\n🎬 Clumping Complete!")
    print(f"   Input events: {len(clumper.events)}")
    print(f"   Output clips: {len(clips)}")
    print(f"   Window size: {args.window}s")
    print(f"   Min intensity: {args.min_intensity}")
    print(f"   Gap threshold: {args.gap_threshold}s")


if __name__ == "__main__":
    main()
