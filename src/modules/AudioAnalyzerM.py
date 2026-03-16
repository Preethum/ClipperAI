"""
AudioAnalyzerM.py - Audio-based combat intensity detector for gaming clips.

Uses the MIT AST (Audio Spectrogram Transformer) model to detect gunfights,
explosions, and other high-intensity combat audio events.

Acts as a fast pre-processor for safe entry/exit detection — when a clear
audio energy spike is found, VLM frame analysis can be skipped entirely.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import numpy as np
from typing import List, Tuple, Optional, Dict

# ── Optional imports (fail gracefully) ────────────────────────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not installed: pip install librosa")

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed: pip install transformers")


# ── Constants ─────────────────────────────────────────────────────────────────

# Labels from AudioSet that indicate combat / gunfight
COMBAT_LABELS = [
    "Gunshot, gunfire",
    "Explosion",
    "Fusillade",
    "Artillery fire",
    "Machine gun",
    # Apex Legends Specific AudioSet mapping (energy weapons, shield breaks)
    "Breaking",
    "Shatter",
    "Smash, crash",
    "Bang",
    "Burst, pop",
]

# AST model — fine-tuned on AudioSet at 10-10-0.4593 mAP
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Audio sample rate required by the model
TARGET_SR = 16_000


class AudioAnalyzer:
    """
    Detects high-intensity combat audio moments in a gameplay video.

    Pipeline:
        1. Extract audio from video via ffmpeg
        2. Compute global RMS energy profile → fast-fail baseline
        3. Slide a chunk window; skip quiet chunks (below 70th percentile RMS)
        4. For loud chunks, classify with AST model on GPU
        5. Sum combat label scores (capped at 1.0); keep events above threshold
        6. Provide entry/exit helpers that find the first energy ramp-up / drop-off
    """

    def __init__(
        self,
        video_path: str,
        chunk_sec: float = 2.0,
        device: int = 0,
        score_threshold: float = 0.30,
    ):
        """
        Args:
            video_path:       Path to the .mp4 (or any video librosa can open).
            chunk_sec:        Length of each audio chunk passed to the model (seconds).
            device:           CUDA device index (0 = first GPU, -1 = CPU).
            score_threshold:  Minimum combined confidence to record an event.
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa is required: pip install librosa")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is required: pip install transformers")

        self.video_path = video_path
        self.chunk_sec = chunk_sec
        self.device = device
        self.score_threshold = score_threshold

        # Loaded on first use to avoid blocking constructor
        self._classifier = None
        self._audio: Optional[np.ndarray] = None   # full waveform (float32, SR=16k)
        self._sr: int = TARGET_SR
        self._rms_profile: Optional[List[Tuple[float, float]]] = None  # (t, rms)

        print(f"[Audio] AudioAnalyzer initialised")
        print(f"   Video   : {video_path}")
        print(f"   Chunk   : {chunk_sec}s  |  Device: {'GPU '+str(device) if device >= 0 else 'CPU'}")
        print(f"   Model   : {AST_MODEL_ID}")
        print(f"   Threshold: {score_threshold}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_classifier(self):
        """Lazy-load the AST model (downloads once, then cached)."""
        if self._classifier is not None:
            return
        print(f"   ⏳ Loading AST model on {'GPU '+str(self.device) if self.device >= 0 else 'CPU'}...")
        self._classifier = hf_pipeline(
            "audio-classification",
            model=AST_MODEL_ID,
            device=self.device,
        )
        print(f"   ✅ AST model loaded")

    def _extract_audio(self) -> np.ndarray:
        """
        Extract audio track from video via ffmpeg into a temp WAV,
        then load with librosa at TARGET_SR.
        """
        if self._audio is not None:
            return self._audio

        print(f"   [Audio] Extracting audio from video...")

        # Find ffmpeg (same helper pattern as gaming_pipeline.py)
        ffmpeg_cmd = _get_ffmpeg_path()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                ffmpeg_cmd, "-y",
                "-i", self.video_path,
                "-vn",                    # no video
                "-ar", str(TARGET_SR),    # sample rate
                "-ac", "1",              # mono
                "-f", "wav",
                tmp_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if result.returncode != 0:
                err = result.stderr.decode(errors="replace")[:300]
                raise RuntimeError(f"ffmpeg audio extraction failed: {err}")

            self._audio, self._sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
            print(f"   ✅ Audio loaded: {len(self._audio)/TARGET_SR:.1f}s at {TARGET_SR}Hz")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return self._audio

    def _build_rms_profile(self, stride_sec: float = 0.5) -> List[Tuple[float, float]]:
        """
        Compute short-term RMS energy across the entire recording.

        Args:
            stride_sec: How often (in seconds) to compute an RMS sample.

        Returns:
            List of (timestamp_sec, rms_value) tuples.
        """
        audio = self._extract_audio()
        stride_samples = int(stride_sec * TARGET_SR)
        window_samples = int(self.chunk_sec * TARGET_SR)

        profile = []
        for start in range(0, len(audio) - window_samples + 1, stride_samples):
            chunk = audio[start : start + window_samples]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            t = start / TARGET_SR + self.chunk_sec / 2  # centre timestamp
            profile.append((t, rms))

        self._rms_profile = profile
        return profile

    def _classify_chunk(self, chunk: np.ndarray) -> float:
        """
        Run AST classifier on a single audio chunk.

        Returns:
            Combined combat score in [0, 1].
        """
        self._load_classifier()
        # The HF pipeline accepts a dict {"raw": array, "sampling_rate": sr}
        results = self._classifier(
            {"raw": chunk, "sampling_rate": TARGET_SR},
            top_k=20,   # get top-20 labels so we catch all combat ones
        )
        # Sum scores for any combat label found
        score = 0.0
        for item in results:
            if item["label"] in COMBAT_LABELS:
                score += item["score"]
        return min(score, 1.0)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_gameplay(self, output_path: Optional[str] = None) -> List[List]:
        """
        Full analysis pass: slide window over audio, classify loud chunks.

        Args:
            output_path: Where to save audio_events.json. Defaults to same
                         directory as the video file.

        Returns:
            List of events in events.json format:
            [[timestamp_float, "audio_combat_spike", score_float], ...]
        """
        audio = self._extract_audio()

        # ── Build global RMS baseline ──────────────────────────────────────────
        print(f"   📊 Building RMS energy profile...")
        rms_profile = self._build_rms_profile(stride_sec=self.chunk_sec / 2)
        all_rms = [rms for _, rms in rms_profile]
        
        # Calculate moving average (e.g., 5-second smoothing)
        window_size = int(5.0 / (self.chunk_sec / 2))  # 5 seconds / stride
        smoothed_rms = np.convolve(all_rms, np.ones(window_size)/window_size, mode='same')
        
        moving_avg_profile = [(rms_profile[i][0], smoothed_rms[i]) for i in range(len(rms_profile))]
        
        # Save smoothed envelope to class instance so external modules (SafeEntryExit) can query it
        self.moving_avg_profile = moving_avg_profile
        
        # Threshold = 70th percentile of SMOOTHED envelope
        rms_threshold = float(np.percentile(smoothed_rms, 70))
        print(f"   📊 Smoothed baseline p70 = {rms_threshold:.6f}  ({len(rms_profile)} windows)")

        # ── Sliding window classification ──────────────────────────────────────
        chunk_samples = int(self.chunk_sec * TARGET_SR)
        stride_samples = chunk_samples  # non-overlapping for speed
        events = []
        total_windows = (len(audio) - chunk_samples) // stride_samples + 1
        skipped = 0

        print(f"   🔍 Scanning {total_windows} chunks ({self.chunk_sec}s each)...")

        for i, start in enumerate(range(0, len(audio) - chunk_samples + 1, stride_samples)):
            chunk = audio[start : start + chunk_samples]
            timestamp = start / TARGET_SR

            # Use the smoothed envelope for fast-failing
            idx = int(timestamp / (self.chunk_sec / 2))
            if idx < len(smoothed_rms) and smoothed_rms[idx] < rms_threshold:
                skipped += 1
                continue

            # GPU inference
            score = self._classify_chunk(chunk)

            if score >= self.score_threshold:
                events.append([round(timestamp, 2), "audio_combat_spike", round(score, 4)])

        print(f"   ✅ Scan complete: {len(events)} combat spikes found ({skipped} quiet chunks skipped)")

        # ── Save JSON output ───────────────────────────────────────────────────
        if output_path is None:
            video_dir = os.path.dirname(os.path.abspath(self.video_path))
            output_path = os.path.join(video_dir, "audio_events.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        print(f"   💾 Saved: {output_path}")

        # ── Save intensity plot ────────────────────────────────────────────────
        plot_path = output_path.replace(".json", "_intensity.png")
        self.save_intensity_plot(moving_avg_profile, events, plot_path)

        return events

    def find_lowest_dip(self, start_sec: float, end_sec: float) -> Optional[float]:
        """
        Finds the exact timestamp of the lowest smoothed RMS energy (quietest moment)
        between two timestamps.
        
        Args:
            start_sec: The beginning of the search window
            end_sec: The end of the search window
            
        Returns:
            The timestamp (float) of the lowest energy dip, or None if no data is available.
        """
        if not hasattr(self, 'moving_avg_profile') or not self.moving_avg_profile:
            return None
            
        # Filter profile to only include points within the requested window
        window_points = [p for p in self.moving_avg_profile if start_sec <= p[0] <= end_sec]
        
        if not window_points:
            return None
            
        # Find the point with the minimum RMS value
        lowest_point = min(window_points, key=lambda p: p[1])
        return lowest_point[0]

    def save_intensity_plot(
        self,
        rms_profile: List[Tuple[float, float]],
        events: List[Dict], # Standardized to allow Dict (from events.json) or List (raw audio)
        output_path: str,
        clips: Optional[List[Dict]] = None,
    ) -> None:
        """
        Save a PNG chart showing RMS audio intensity over time, with detected
        combat spikes overlaid as red vertical markers, and final clips shaded green.

        Args:
            rms_profile:  List of (timestamp, rms_energy) from _build_rms_profile.
            events:       Detected combat events [[timestamp, label, score], ...].
            output_path:  Where to save the PNG.
            clips:        Optional list of clip dicts (from clips.json) to shade.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend — no GUI window
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            print("   ⚠️  matplotlib not installed, skipping plot (pip install matplotlib)")
            return

        if not rms_profile:
            print("   ⚠️  No RMS profile data, skipping plot")
            return

        times = [t for t, _ in rms_profile]
        rms_vals = [r for _, r in rms_profile]
        
        # Standardize events to dicts if they are lists (from old API)
        std_events = []
        for e in events:
            if isinstance(e, dict):
                std_events.append(e)
            elif isinstance(e, list) and len(e) >= 3:
                std_events.append({"timestamp_s": e[0], "label": e[1], "confidence": e[2]})

        spike_ts = [e["timestamp_s"] for e in std_events if e.get("label") == "audio_combat_spike"]
        spike_scores = [e["confidence"] for e in std_events if e.get("label") == "audio_combat_spike"]
        
        # Visual markers
        eliminated_ts = [e["timestamp_s"] for e in std_events if e.get("label") in ("Squad Eliminated", "ELIMINATED Text")]
        combat_feed_ts = [e["timestamp_s"] for e in std_events if e.get("label") in ("Combat Feed / Kill Feed Updates", "Knockdown/Kill Notifications")]
        damage_ts = [e["timestamp_s"] for e in std_events if e.get("label") in ("Floating Damage Numbers", "loating Damage Numbers", "Directional Damage Indicators")]
        victory_ts = [e["timestamp_s"] for e in std_events if e.get("label") == "Victory"]

        # ── Convert seconds → MM:SS labels ───────────────────────────────────
        def fmt_time(x, _):
            m, s = divmod(int(x), 60)
            return f"{m}:{s:02d}"

        # ── Build figure ──────────────────────────────────────────────────────
        fig, (ax_rms, ax_score) = plt.subplots(
            2, 1, figsize=(20, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.patch.set_facecolor("#0e1117")
        for ax in (ax_rms, ax_score):
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="#cdd6f4", labelsize=8)
            ax.xaxis.label.set_color("#cdd6f4")
            ax.yaxis.label.set_color("#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#313244")

        # ── Top panel: RMS waveform ───────────────────────────────────────────
        ax_rms.fill_between(times, rms_vals, alpha=0.35, color="#89b4fa")
        ax_rms.plot(times, rms_vals, color="#cba6f7", linewidth=1.5, label="Smoothed RMS Envelope")

        # 70th-percentile threshold line
        p70 = float(np.percentile(rms_vals, 70))
        ax_rms.axhline(p70, color="#f38ba8", linewidth=1.0, linestyle="--", alpha=0.9,
                       label=f"Smoothed p70 ({p70:.4f})")

        # Red vertical lines at each spike
        for ts in spike_ts:
            ax_rms.axvline(ts, color="#f38ba8", linewidth=0.6, alpha=0.5)

        ax_rms.set_ylabel("RMS Energy", fontsize=9)
        ax_rms.set_title(
            f"[Audio] Audio Combat Intensity — {os.path.basename(self.video_path)}",
            color="#cdd6f4", fontsize=11, pad=10
        )
        ax_rms.legend(loc="upper right", fontsize=7, facecolor="#1e1e2e", labelcolor="#cdd6f4")

        # ── Bottom panel: spike confidence scores ─────────────────────────────
        if spike_ts:
            ax_score.bar(
                spike_ts, spike_scores,
                width=max(2.0, self.chunk_sec),
                color="#f38ba8", alpha=0.7, label="Audio Confidence"
            )
            
        # Overlay visual markers at the top of ax_score
        if victory_ts:
            ax_score.scatter(victory_ts, [1.20] * len(victory_ts), color="#f9e2af", marker="*", s=150, label="Victory", zorder=6, edgecolors="white", linewidth=0.5)
        if eliminated_ts:
            ax_score.scatter(eliminated_ts, [1.15] * len(eliminated_ts), color="#f38ba8", marker="*", s=80, label="Squad Eliminated", zorder=5)
        if combat_feed_ts:
            ax_score.scatter(combat_feed_ts, [1.10] * len(combat_feed_ts), color="#fab387", marker="v", s=30, label="Combat Feed", zorder=5)
        if damage_ts:
            ax_score.scatter(damage_ts, [1.05] * len(damage_ts), color="#f9e2af", marker=".", s=20, label="Damage Numbers", zorder=5)

        # ── Overlay final video clips (if provided) ───────────────────────────
        if clips:
            for i, clip in enumerate(clips):
                c_start = clip.get('safe_entry', {}).get('timestamp', clip['start_time'])
                c_end = clip.get('safe_exit', {}).get('timestamp', clip['end_time'])
                
                # Shade the clip region in green
                ax_rms.axvspan(c_start, c_end, color="#a6e3a1", alpha=0.2, lw=0)
                ax_score.axvspan(c_start, c_end, color="#a6e3a1", alpha=0.2, lw=0)
                
                # Explicit vertical lines for start and end
                ax_rms.axvline(c_start, color="#a6e3a1", linewidth=2.0, linestyle="--")
                ax_rms.axvline(c_end, color="#f9e2af", linewidth=2.0, linestyle="--")
                ax_score.axvline(c_start, color="#a6e3a1", linewidth=2.0, linestyle="--")
                ax_score.axvline(c_end, color="#f9e2af", linewidth=2.0, linestyle="--")
                
                # Add text label for the clip
                ax_rms.text(
                    c_start, ax_rms.get_ylim()[1] * 0.95, f"Clip {i+1} Start", 
                    color="#a6e3a1", fontsize=8, fontweight="bold", 
                    va="top", ha="left", rotation=90, 
                    bbox=dict(facecolor="#1e1e2e", alpha=0.7, edgecolor="none", pad=1.0)
                )
                
                ax_rms.text(
                    c_end, ax_rms.get_ylim()[1] * 0.95, f"Clip {i+1} End", 
                    color="#f9e2af", fontsize=8, fontweight="bold", 
                    va="top", ha="right", rotation=90, 
                    bbox=dict(facecolor="#1e1e2e", alpha=0.7, edgecolor="none", pad=1.0)
                )

        ax_score.set_ylim(0, 1.25)  # Expanded to fit visual markers
        ax_score.set_ylabel("Score", fontsize=9)
        ax_score.set_xlabel("Time", fontsize=9)
        ax_score.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
        ax_score.xaxis.set_major_locator(ticker.MultipleLocator(60))  # tick every 60s
        ax_score.legend(loc="upper right", fontsize=7, facecolor="#1e1e2e", labelcolor="#cdd6f4", ncol=4)

        # ── Footer stats ──────────────────────────────────────────────────────
        total_min = times[-1] / 60 if times else 0
        fig.text(
            0.01, 0.01,
            f"Duration: {total_min:.1f} min  |  Spikes: {len(spike_ts)}  |  Threshold score: {self.score_threshold}",
            color="#6c7086", fontsize=7
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"   📊 Plot saved: {output_path}")

    def get_audio_intensity_profile(
        self, start_time: float, end_time: float, resolution: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Return the raw RMS energy profile within a time window.
        Used for plotting — NOT for entry/exit detection (use get_combat_score_profile instead).

        Args:
            start_time:  Window start (seconds).
            end_time:    Window end (seconds).
            resolution:  Time step between samples (seconds).

        Returns:
            List of (timestamp, rms_energy) within [start_time, end_time].
        """
        audio = self._extract_audio()
        window_samples = max(1, int(self.chunk_sec * TARGET_SR))
        stride_samples = max(1, int(resolution * TARGET_SR))

        start_sample = int(start_time * TARGET_SR)
        end_sample = min(int(end_time * TARGET_SR), len(audio))

        profile = []
        for pos in range(start_sample, end_sample - window_samples + 1, stride_samples):
            chunk = audio[pos : pos + window_samples]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            t = pos / TARGET_SR + self.chunk_sec / 2
            profile.append((t, rms))

        return profile

    def get_combat_score_profile(
        self, start_time: float, end_time: float, resolution: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Return the AST combat-score profile within a time window.

        Unlike get_audio_intensity_profile (raw RMS), this classifies each chunk
        with the AST model and returns the COMBAT LABEL score as the intensity.
        This means only gunshots, explosions, etc. contribute to the profile —
        ambient music or loud non-combat sounds are effectively ignored.

        A fast-fail RMS gate skips classifying very quiet chunks to save GPU time.

        Args:
            start_time:  Window start (seconds).
            end_time:    Window end (seconds).
            resolution:  Stride between windows (seconds). Smaller = more accurate
                         but more classifier calls. 0.5s gives 2 calls/sec.

        Returns:
            List of (timestamp, combat_score) where score is in [0, 1].
        """
        audio = self._extract_audio()
        window_samples = max(1, int(self.chunk_sec * TARGET_SR))
        stride_samples = max(1, int(resolution * TARGET_SR))

        start_sample = int(start_time * TARGET_SR)
        end_sample = min(int(end_time * TARGET_SR), len(audio))

        # Compute a quick RMS gate threshold from this window
        all_rms = []
        for pos in range(start_sample, end_sample - window_samples + 1, stride_samples):
            chunk = audio[pos : pos + window_samples]
            all_rms.append(float(np.sqrt(np.mean(chunk ** 2))))

        if not all_rms:
            return []

        rms_gate = float(np.percentile(all_rms, 30))  # skip the quietest 30%

        profile = []
        for i, pos in enumerate(range(start_sample, end_sample - window_samples + 1, stride_samples)):
            chunk = audio[pos : pos + window_samples]
            t = pos / TARGET_SR + self.chunk_sec / 2
            local_rms = all_rms[i]

            if local_rms < rms_gate:
                # Too quiet — no combat possible, score=0
                profile.append((t, 0.0))
            else:
                # Run AST classifier → get combat score
                score = self._classify_chunk(chunk)
                profile.append((t, score))

        return profile

    def find_audio_entry_point(
        self, window_start: float, window_end: float, clip_start: Optional[float] = None
    ) -> Tuple[Optional[float], float]:
        """
        Find where combat audio first ramps up sharply (fight begins).

        Uses the COMBAT SCORE profile (AST classifier), not raw RMS.
        This means only gunshots/explosions trigger an onset — ambient music
        and loud non-combat sounds are ignored.

        Args:
            window_start: Search window start (seconds).
            window_end:   Search window end (seconds).
            clip_start:   The original unpadded clip start time (limit search up to this point).

        Returns:
            (timestamp, confidence) — timestamp is None if no clear onset found.
        """
        if clip_start is not None:
            # Don't search for entry points *after* the core clip event has already started
            window_end = min(window_end, clip_start)

        profile = self.get_combat_score_profile(window_start, window_end, resolution=0.5)
        if len(profile) < 4:
            return None, 0.0

        times = np.array([t for t, _ in profile])
        scores = np.array([s for _, s in profile])

        # 1. Find a "Combat Anchor" (the latest high-confidence combat signal)
        # We search backwards from the end because we want the FIGHT related to the clip
        anchor_idx = -1
        # Minimum combat score to be considered the starting "anchor"
        anchor_threshold = max(self.score_threshold * 0.5, 0.1) 
        
        for i in range(len(scores) - 1, -1, -1):
            if scores[i] >= anchor_threshold:
                anchor_idx = i
                break
        
        if anchor_idx == -1:
            # No combat found in window — fallback to raw RMS activity if any
            return None, 0.0

        anchor_time = float(times[anchor_idx])
        confidence = float(scores[anchor_idx])

        # 2. Trace back for the "buildup" (RMS lead-up)
        # We look for where the raw noise started, allowing for small gaps
        intensity_profile = self.get_audio_intensity_profile(window_start, anchor_time, resolution=0.2)
        if not intensity_profile:
            return anchor_time, confidence

        rms_vals = [r for _, r in intensity_profile]
        noise_floor = np.percentile(rms_vals, 20)
        rms_thresh = max(noise_floor * 2.0, 0.005)
        
        onset_time = anchor_time
        silence_count = 0
        gap_limit = 2.0 # Allow 2s of silence within a buildup (e.g. reload, sneaker)
        
        # Iterate backwards
        for t_rms, v_rms in reversed(intensity_profile):
            if v_rms > rms_thresh:
                onset_time = t_rms
                silence_count = 0
            else:
                silence_count += 0.2 # resolution
                if silence_count > gap_limit:
                    break # Hard stop — real silence found
        
        if onset_time < anchor_time:
            print(f"   [Audio] Lead-up found: anchored at {anchor_time:.1f}s, tracing back to {onset_time:.1f}s")
            confidence = max(confidence, 0.7) # boost confidence

        return onset_time, float(np.clip(confidence, 0.0, 1.0))

    def find_audio_exit_point(
        self, window_start: float, window_end: float, clip_end: Optional[float] = None
    ) -> Tuple[Optional[float], float]:
        """
        Find where combat audio drops off sharply (fight ends).

        Uses the COMBAT SCORE profile (AST classifier), not raw RMS.
        Searches for the sharpest drop in combat score after the peak,
        ignoring non-combat sounds entirely.

        Args:
            window_start: Search window start (seconds).
            window_end:   Search window end (seconds).
            clip_end:     The original unpadded clip end time (limit search after this point).

        Returns:
            (timestamp, confidence) — timestamp is None if no clear offset found.
        """
        if clip_end is not None:
            # Don't search for exit points *before* the core clip event has finished
            window_start = max(window_start, clip_end)

        profile = self.get_combat_score_profile(window_start, window_end, resolution=0.5)
        if len(profile) < 4:
            return None, 0.0

        times = np.array([t for t, _ in profile])
        scores = np.array([s for _, s in profile])

        if np.max(scores) < 0.1:
            return None, 0.0

        # 1. Find the "Combat Anchor" (the last combat spike in this window)
        # We start searching from the peak combat moment onwards
        peak_idx = int(np.argmax(scores))
        anchor_idx = peak_idx
        
        # Look for any combat above threshold after the peak
        anchor_threshold = max(self.score_threshold * 0.5, 0.1)
        for i in range(len(scores) - 1, peak_idx, -1):
            if scores[i] >= anchor_threshold:
                anchor_idx = i
                break
        
        anchor_time = float(times[anchor_idx])
        confidence = float(scores[anchor_idx])

        # 2. Trace forward for the "cooldown" (RMS trace-forward)
        # We look for where the raw noise finally dies down
        intensity_profile = self.get_audio_intensity_profile(anchor_time, window_end, resolution=0.2)
        if not intensity_profile:
            return anchor_time, confidence

        rms_vals = [r for _, r in intensity_profile]
        noise_floor = np.percentile(rms_vals, 20)
        rms_thresh = max(noise_floor * 2.5, 0.005) # Sidelined ambient noise
        
        exit_time = anchor_time
        silence_count = 0
        gap_limit = 3.0 # Allow 3s of celebration/looting/dialogue gaps
        
        # Iterate forwards
        for t_rms, v_rms in intensity_profile:
            if v_rms > rms_thresh:
                exit_time = t_rms
                silence_count = 0
            else:
                silence_count += 0.2 # resolution
                if silence_count > gap_limit:
                    break # Hard stop — fight is definitely over
        
        if exit_time > anchor_time:
            print(f"   [Audio] Cooldown found: anchored at {anchor_time:.1f}s, tracing forward to {exit_time:.1f}s")
            confidence = max(confidence, 0.7)

        return exit_time, float(np.clip(confidence, 0.0, 1.0))


# ── Path helper ────────────────────────────────────────────────────────────────

def _get_ffmpeg_path(binary: str = "ffmpeg") -> str:
    """Locate ffmpeg binary — project bin/ first, then system PATH."""
    # Project root is two levels up from src/modules/
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(module_dir))
    bin_path = os.path.join(project_root, "bin", f"{binary}.exe")
    if os.path.isfile(bin_path):
        return bin_path
    found = shutil.which(binary)
    if found:
        return found
    raise FileNotFoundError(
        f"{binary} not found. Add it to {project_root}/bin/ or ensure it's on PATH."
    )


# ── shutil import was moved to the top ────────────────────────────────────────


# ── Standalone test ─────────────────────────────────────────────────────────--

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AudioAnalyzerM — standalone audio combat detector"
    )
    parser.add_argument("video", help="Path to gameplay video (.mp4)")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for audio_events.json (default: same directory as video)",
    )
    parser.add_argument(
        "--chunk", type=float, default=2.0,
        help="Chunk size in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index: 0=GPU0, -1=CPU (default: 0)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Minimum combat score to record event (default: 0.30)",
    )
    parser.add_argument(
        "--test_window",
        nargs=2, type=float, metavar=("START", "END"),
        help="Also test entry/exit detection on a time window (e.g. --test_window 60 120)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AudioAnalyzerM — standalone test")
    print("=" * 60)

    analyzer = AudioAnalyzer(
        video_path=args.video,
        chunk_sec=args.chunk,
        device=args.device,
        score_threshold=args.threshold,
    )

    # Full gameplay scan
    events = analyzer.process_gameplay(output_path=args.output)

    print(f"\n📋 Detected {len(events)} combat audio events:")
    for ts, label, score in events[:20]:  # Show first 20
        mins = int(ts // 60)
        secs = ts % 60
        print(f"   {mins:02d}:{secs:05.2f}  score={score:.3f}  [{label}]")
    if len(events) > 20:
        print(f"   ... and {len(events) - 20} more")

    # Optional window test
    if args.test_window:
        w_start, w_end = args.test_window
        print(f"\n🔍 Testing entry/exit point detection in [{w_start:.0f}s – {w_end:.0f}s]...")

        entry_ts, entry_conf = analyzer.find_audio_entry_point(w_start, w_end)
        exit_ts,  exit_conf  = analyzer.find_audio_exit_point(w_start, w_end)

        print(f"   Entry: {entry_ts}s  (confidence={entry_conf:.2f})")
        print(f"   Exit : {exit_ts}s  (confidence={exit_conf:.2f})")
