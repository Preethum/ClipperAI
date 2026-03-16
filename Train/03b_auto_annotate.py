import sys
import os

# IMPORTANT DECORATOR FIX FOR WINDOWS [WinError 1114]
# Qt6 and PyTorch/CUDA both use heavily modified C++ redistributable runtimes.
# PyTorch MUST be imported into memory *before* PyQt6/cv2 to bind its core DLLs.
from ultralytics import YOLO
import torch 

import cv2
import numpy as np
import subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
                             QDoubleSpinBox, QGroupBox, QProgressDialog, QMessageBox,
                             QLineEdit, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# Context paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORT_DIR = os.path.join(BASE_DIR, "import_to_label_studio")

class ProxyWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self._is_cancelled = False

    def run(self):
        try:
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Could not open source video.")
                return

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Output path
            base, ext = os.path.splitext(self.video_path)
            proxy_path = f"{base}_proxy{ext}"
            
            # Use a fast codec (XVID or MP4V)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(proxy_path, fourcc, fps, (width, height))

            frame_idx = 0
            while True:
                if self._is_cancelled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO Inference
                results = model.predict(source=frame, verbose=False)
                r = results[0]
                
                # Draw boxes (baked-in)
                plotted_frame = r.plot(font_size=2, line_width=2)
                
                # Write to proxy
                writer.write(plotted_frame)
                
                frame_idx += 1
                if frame_idx % 5 == 0:
                    self.progress.emit(int((frame_idx / total_frames) * 100))

            cap.release()
            writer.release()

            if self._is_cancelled:
                if os.path.exists(proxy_path):
                    os.remove(proxy_path)
                self.error.emit("Proxy generation cancelled.")
            else:
                self.finished.emit(proxy_path)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._is_cancelled = True

class AutoAnnotateApp(QMainWindow):
    def __init__(self, video_path=None, model_path=None):
        super().__init__()
        
        self.setWindowTitle("ClipperAI - Active Learning Scanner")
        self.setGeometry(100, 100, 1280, 800)
        
        # Core State
        self.video_path = video_path
        self.proxy_path = None
        self.model_path = model_path
        self.model = None
        
        # We now keep track of TWO video captures
        self.cap_source = None  # Original high-res video (for saving pristine frames)
        self.cap_display = None # Proxy video or source video (for smooth scrubbing)
        
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.session_id = "manual_scan"
        self.session_tag = ""
        self.saved_count = 0
        
        # Initialize UI immediately so elements exist before loading anything
        self.init_ui()

        # Try to find defaults if not provided
        if not self.video_path or not self.model_path:
            self.discover_defaults()

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Enforce video & model selection at startup
        QTimer.singleShot(100, self.check_initial_inputs)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # --- Top Bar (Controls & Config) ---
        top_layout = QHBoxLayout()
        
        self.btn_load_vid = QPushButton("Load Source Video")
        self.btn_load_vid.clicked.connect(self.open_video_dialog)
        top_layout.addWidget(self.btn_load_vid)
        
        self.btn_load_proxy = QPushButton("Load Annotated Video")
        self.btn_load_proxy.clicked.connect(self.open_proxy_dialog)
        self.btn_load_proxy.setStyleSheet("background-color: #673AB7; color: white;")
        top_layout.addWidget(self.btn_load_proxy)
        
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.open_model_dialog)
        top_layout.addWidget(self.btn_load_model)

        self.btn_preload = QPushButton("Preload Detections (Create Proxy)")
        self.btn_preload.clicked.connect(self.start_proxy_gen)
        self.btn_preload.setStyleSheet("background-color: #E91E63; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_preload)
        
        self.chk_live_ai = QCheckBox("Enable Live AI Scanning")
        self.chk_live_ai.setChecked(True) # Default to true since proxy is gone
        top_layout.addWidget(self.chk_live_ai)
        
        self.txt_session_tag = QLineEdit()
        self.txt_session_tag.setPlaceholderText("Optional Scan Tag")
        top_layout.addWidget(self.txt_session_tag)
        
        # Confidence Settings
        conf_group = QGroupBox("Uncertainty Range")
        conf_layout = QHBoxLayout()
        
        conf_layout.addWidget(QLabel("Low:"))
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setRange(0.01, 0.99)
        self.spin_low.setSingleStep(0.05)
        self.spin_low.setValue(0.20)
        conf_layout.addWidget(self.spin_low)
        
        conf_layout.addWidget(QLabel("High:"))
        self.spin_high = QDoubleSpinBox()
        self.spin_high.setRange(0.01, 1.0)
        self.spin_high.setSingleStep(0.05)
        self.spin_high.setValue(0.75)
        conf_layout.addWidget(self.spin_high)
        
        conf_group.setLayout(conf_layout)
        top_layout.addWidget(conf_group)
        
        self.lbl_status = QLabel("Ready")
        top_layout.addWidget(self.lbl_status)
        top_layout.addStretch()
        
        layout.addLayout(top_layout)
        
        # --- Video Display Area ---
        self.lbl_video = QLabel("No Video Loaded")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        # Make the video label expand to fill space
        self.lbl_video.setMinimumSize(640, 360)
        layout.addWidget(self.lbl_video, stretch=1)
        
        # --- Timeline Scrubber ---
        scrub_layout = QHBoxLayout()
        self.lbl_time = QLabel("0 / 0")
        scrub_layout.addWidget(self.lbl_time)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        # Fast scrubbing: only update frame without AI while dragging
        self.slider.sliderMoved.connect(self.fast_scrub_video)
        # Full update: run AI when slider is released
        self.slider.sliderReleased.connect(self.seek_video_final)
        scrub_layout.addWidget(self.slider)
        
        layout.addLayout(scrub_layout)
        
        # --- Bottom Bar (Action Buttons) ---
        bot_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("Play/Pause")
        self.btn_play.clicked.connect(self.toggle_playback)
        bot_layout.addWidget(self.btn_play)
        
        # Playback speed gauge
        bot_layout.addWidget(QLabel("Speed:"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 10.0)
        self.spin_speed.setSingleStep(0.25)
        self.spin_speed.setValue(1.0)
        self.spin_speed.valueChanged.connect(self.update_playback_speed)
        bot_layout.addWidget(self.spin_speed)
        
        self.btn_scan = QPushButton("Scan Forward to Next Uncertain Frame")
        self.btn_scan.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_scan.clicked.connect(self.scan_to_next_uncertain)
        bot_layout.addWidget(self.btn_scan)
        
        self.btn_save = QPushButton("💾 SAVE Current Frame")
        self.btn_save.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_current_frame)
        bot_layout.addWidget(self.btn_save)
        
        bot_layout.addStretch()
        self.lbl_saved = QLabel("Saved: 0")
        bot_layout.addWidget(self.lbl_saved)
        
        layout.addLayout(bot_layout)

    def discover_defaults(self):
        """Look for standard project files if none provided."""
        # 1. Search for Model
        if not self.model_path:
            # Prefer 'best.pt' if in a trained dir, else highest nanoyolo or standard n
            candidates = [
                os.path.join(BASE_DIR, "..", "yolov8m.pt"),
                os.path.join(BASE_DIR, "..", "yolov8n.pt"),
                os.path.join(BASE_DIR, "..", "yolo26n.pt")
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.model_path = os.path.abspath(c)
                    break
                    
        # 2. Search for Video
        if not self.video_path:
            # Look for common video files in root
            root = os.path.abspath(os.path.join(BASE_DIR, ".."))
            for f in os.listdir(root):
                if f.endswith((".mp4", ".mkv")) and not "annotated" in f.lower() and not "proxy" in f.lower():
                    # Pick the first one for now (or latest)
                    self.video_path = os.path.join(root, f)
                    break

    def check_initial_inputs(self):
        # 1. Load Video if we have a path
        if self.video_path and os.path.exists(self.video_path):
            self.load_video(self.video_path)
            self.lbl_status.setText(f"Auto-loaded video: {os.path.basename(self.video_path)}")
        else:
            QMessageBox.information(self, "Welcome", "Please select a gameplay video to begin.")
            self.open_video_dialog()
            
        # 2. Load Model if we have a path
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
            self.lbl_status.setText(f"Auto-loaded model: {os.path.basename(self.model_path)}")
        else:
            QMessageBox.information(self, "Model Required", "Please select a YOLO model (.pt).")
            self.open_model_dialog()

    def load_model(self, path):
        try:
            self.lbl_status.setText(f"Loading {os.path.basename(path)}...")
            QApplication.processEvents()
            self.model = YOLO(path)
            self.model_path = path
            self.lbl_status.setText(f"Loaded: {os.path.basename(path)}")
            self.btn_preload.setEnabled(True)
        except Exception as e:
            self.lbl_status.setText(f"Error loading model: {str(e)}")

    def load_video(self, path):
        if self.cap_source:
            self.cap_source.release()
        if self.cap_display:
            self.cap_display.release()
            
        self.cap_source = cv2.VideoCapture(path)
        self.cap_display = cv2.VideoCapture(path) # Default to using source for display
        
        if not self.cap_source.isOpened():
            self.lbl_status.setText("Failed to open source video.")
            return
            
        self.video_path = path
        self.proxy_path = None
        self.total_frames = int(self.cap_source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.total_frames - 1)
        self.lbl_status.setText(f"Loaded: {os.path.basename(path)}")
        self.btn_preload.setEnabled(True)

        # AUTOMATIC PROXY DETECTION
        # Check if a proxy video already exists in the same directory
        base, ext = os.path.splitext(path)
        potential_proxy = f"{base}_proxy{ext}"
        if os.path.exists(potential_proxy):
            self.load_proxy(potential_proxy)
            self.lbl_status.setText(f"Loaded: {os.path.basename(path)} (Proxy Auto-detected)")
        
        # Generate session ID from filename
        base = os.path.basename(path)
        self.session_id = os.path.splitext(base)[0].replace(" ", "_")
        self.lbl_status.setText(f"Video: {base}")
            
        self.seek_video(0)

    def open_video_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", BASE_DIR, "Videos (*.mp4 *.mkv *.avi)")
        if file_name:
            self.load_video(file_name)
            
    def open_model_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open YOLO Model", BASE_DIR, "PyTorch Models (*.pt)")
        if file_name:
            self.load_model(file_name)

    def open_proxy_dialog(self):
        if not self.video_path:
            QMessageBox.warning(self, "Load Source First", "Please load a Source Video before loading an annotated proxy.")
            return
            
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Proxy Video", os.path.dirname(self.video_path), "Videos (*.mp4 *.mkv *.avi)")
        if file_name:
            # Check if this proxy matches the source video's frame count
            test_cap = cv2.VideoCapture(file_name)
            proxy_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_cap.release()
            
            # Allow a margin of error for OpenCV decoders (5%)
            if abs(proxy_frames - self.total_frames) > (self.total_frames * 0.05):
                reply = QMessageBox.question(self, "Frame Count Mismatch", 
                    f"The proxy video has {proxy_frames} frames but the source has {self.total_frames}. "
                    "This will cause the bounding boxes to sync incorrectly. Proceed anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            
            self.load_proxy(file_name)

    def load_proxy(self, proxy_path):
        if self.cap_display and self.cap_display != self.cap_source:
             self.cap_display.release()
             
        self.cap_display = cv2.VideoCapture(proxy_path)
        if not self.cap_display.isOpened():
            self.lbl_status.setText("Failed to open Proxy video.")
            self.cap_display = cv2.VideoCapture(self.video_path) # fallback
            return
            
        self.proxy_path = proxy_path
        self.lbl_status.setText(f"Proxy Active: {os.path.basename(proxy_path)}")
        self.seek_video(self.current_frame_idx)

    def start_proxy_gen(self):
        """Starts the background thread to generate a proxy video."""
        if not self.video_path or not self.model_path:
            QMessageBox.warning(self, "Missing Files", "Please load BOTH a video and a model first.")
            return

        reply = QMessageBox.question(self, "Start Preloading?", 
            "This will scan the entire video and bake detections into a proxy file for smooth scrubbing. "
            "This may take a few minutes depending on GPU speed. Start?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.No:
            return

        # Setup Progress Dialog
        self.progress_diag = QProgressDialog("Generating Proxy Video...", "Cancel", 0, 100, self)
        self.progress_diag.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_diag.setAutoClose(True)
        self.progress_diag.setMinimumDuration(0)

        # Setup Worker
        self.worker = ProxyWorker(self.video_path, self.model_path)
        self.worker.progress.connect(self.progress_diag.setValue)
        self.worker.finished.connect(self.on_proxy_finished)
        self.worker.error.connect(self.on_proxy_error)
        
        # Handle manual cancel
        self.progress_diag.canceled.connect(self.worker.cancel)
        
        self.worker.start()
        self.btn_preload.setEnabled(False)

    def on_proxy_finished(self, proxy_path):
        self.btn_preload.setEnabled(True)
        QMessageBox.information(self, "Success", f"Proxy video generated successfully!\n{os.path.basename(proxy_path)}")
        self.load_proxy(proxy_path)

    def on_proxy_error(self, error_msg):
        self.btn_preload.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to generate proxy:\n{error_msg}")


    def toggle_playback(self):
        if not self.cap_display:
            return
            
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
        else:
            speed = self.spin_speed.value()
            interval_ms = int(33 / speed)
            if interval_ms < 1: interval_ms = 1
            self.timer.start(interval_ms)
            self.is_playing = True

    def update_playback_speed(self):
        if self.is_playing:
            speed = self.spin_speed.value()
            interval_ms = int(33 / speed)
            if interval_ms < 1: interval_ms = 1
            self.timer.start(interval_ms)

    def fast_scrub_video(self, frame_idx):
        if not self.cap_display:
            return
            
        self.current_frame_idx = frame_idx
        self.cap_display.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        # Skip YOLO during fast drag so it doesn't lag. If proxy is active, it's skipped anyway.
        self.update_frame_display(skip_yolo=True)
        
    def seek_video_final(self):
        # Called when the user releases the slider
        # If proxy is active, NEVER run YOLO because it's baked into the video
        skip = True if self.proxy_path else False
        self.update_frame_display(skip_yolo=skip)

    def seek_video(self, frame_idx):
        if not self.cap_display:
            return
            
        self.current_frame_idx = frame_idx
        self.cap_display.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        skip = True if self.proxy_path else False
        self.update_frame_display(skip_yolo=skip)

    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.update_frame_display()
        else:
            self.toggle_playback() # Stop at end

    def update_frame_display(self, return_results=False, skip_yolo=False):
        """Reads the current frame, runs YOLO, and displays the result."""
        if not self.cap_display:
            return None
            
        ret, frame = self.cap_display.read()
        if not ret:
            return None
            
        # Update Scrubber
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self.lbl_time.setText(f"{self.current_frame_idx} / {self.total_frames}")
        
        # Pull the absolute pristine frame from the SOURCE video in the background for saving
        # so we never accidentally save a compressed fuzzy proxy frame to the dataset
        # OPTIMIZATION: We ONLY seek the source when actually saving now, to keep playback 60fps smooth.
        self.last_raw_frame = None # Reset every frame to ensure we grab fresh on Save
        
        # Run YOLO Inference
        r = None
        is_uncertain = False
        display_img = frame.copy()
        
        # Determine if we should skip YOLO:
        # 1. If we have a proxy video, NEVER run YOLO (boxes are baked in).
        # 2. We MUST run YOLO if the user is playing/scrubbing in "Live AI" mode, 
        #    OR if we are forcing it (return_results).
        if getattr(self, 'proxy_path', None):
             skip_yolo = True
        elif skip_yolo:
             # explicitly passed as True (e.g. fast scrubbing)
             pass
        elif not getattr(self, 'chk_live_ai', None) or not self.chk_live_ai.isChecked():
             if not return_results: # Still allow forced scans (Scan Forward)
                 skip_yolo = True
                 
        if getattr(self, 'model', None) and not skip_yolo:
            results = self.model.predict(source=frame, verbose=False)
            r = results[0]
            self.last_yolo_result = r
            
            low = self.spin_low.value()
            high = self.spin_high.value()
            
            if r.boxes:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if low <= conf <= high:
                        is_uncertain = True
                        break
                        
            # Draw YOLO boxes on the display image
            display_img = r.plot(font_size=2, line_width=1)
            
        # Add visual indicator if uncertain
        if is_uncertain:
            cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), (0, 165, 255), 4) # Orange border
            cv2.putText(display_img, "UNCERTAIN DETECTIONS FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Convert OpenCV BGR to Qt RGB format
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        
        # Important: QImage needs a reference to the data, so keep rgb_img alive natively or pass a copy block
        self.current_qimage = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(self.current_qimage)
        
        # Keep aspect ratio when scaling to window
        scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)
        
        if return_results:
            return is_uncertain
            
    def scan_to_next_uncertain(self):
        """Fast-forwards until it finds an uncertain frame."""
        if not self.cap_display or not self.model:
            self.lbl_status.setText("Load a video and model first!")
            return
            
        self.lbl_status.setText("Scanning forward...")
        QApplication.processEvents()
        
        # Pause playback if running
        if self.is_playing:
            self.toggle_playback()
            
        found = False
        while self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 5  # Skip frames to scan faster
            self.cap_display.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            
            # Force run YOLO on source frame to determine uncertainty
            if self.cap_source:
                self.cap_source.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = self.cap_source.read()
                if ret and self.model:
                    results = self.model.predict(source=frame, verbose=False)
                    r = results[0]
                    low = self.spin_low.value()
                    high = self.spin_high.value()
                    
                    if r.boxes:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            if low <= conf <= high:
                                found = True
                                break
                
            if found:
                break
                
            # Allow GUI to update occasionally so it doesn't totally freeze
            if self.current_frame_idx % 30 == 0:
                QApplication.processEvents()
                
        if found:
            self.lbl_status.setText(f"Stopped at uncertain frame: {self.current_frame_idx}")
            # Ensure the frame display actually updates the UI one last time
            self.cap_display.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self.update_frame_display(skip_yolo=False)
        else:
            self.lbl_status.setText("Reached end of video. No more uncertain frames found.")

    def save_current_frame(self):
        """Saves pristine frame to Label Studio dir."""
        # 1. Explicitly seek and read from SOURCE video for maximum quality
        if not self.cap_source:
            self.lbl_status.setText("No source video to save from!")
            return
            
        self.cap_source.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, raw_frame = self.cap_source.read()
        if not ret:
            self.lbl_status.setText("Failed to read high-res frame!")
            return
            
        self.last_raw_frame = raw_frame.copy()
        
        os.makedirs(IMPORT_DIR, exist_ok=True)
        
        # Combine session ID, tag, and frame number
        tag = self.txt_session_tag.text().strip() if getattr(self, 'txt_session_tag', None) else ""
        safe_tag = "".join(c for c in tag if c.isalnum() or c in ('-', '_'))
        
        if safe_tag:
            base_name = f"{self.session_id}_{safe_tag}_f{self.current_frame_idx:06d}"
        else:
            base_name = f"{self.session_id}_f{self.current_frame_idx:06d}"
            
        img_path = os.path.join(IMPORT_DIR, f"{base_name}.jpg")
        
        # Save pristine image
        cv2.imwrite(img_path, self.last_raw_frame)
            
        self.saved_count += 1
        self.lbl_saved.setText(f"Saved: {self.saved_count}")
        self.lbl_status.setText(f"Saved High-Res Original Frame: {base_name}.jpg")
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale current frame if window resizes
        if self.cap_display and getattr(self, 'current_qimage', None) is not None:
             pixmap = QPixmap.fromImage(self.current_qimage)
             scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
             self.lbl_video.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """Allow scrolling through frames using the mouse wheel."""
        if not self.cap_display:
            return
            
        # Stop playback if playing
        if getattr(self, 'is_playing', False):
            self.toggle_playback()
            
        # Scroll up = forward, scroll down = backward
        delta = event.angleDelta().y()
        if delta > 0:
            new_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
        else:
            new_idx = max(0, self.current_frame_idx - 1)
            
        if new_idx != self.current_frame_idx:
            self.seek_video(new_idx)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Active Learning Interactive GUI")
    parser.add_argument("--video", type=str, default="", help="Path to initial video")
    parser.add_argument("--model", type=str, default="", help="Path to YOLO model")
    args, unknown = parser.parse_known_args() # Ignore stray arguments for PyQt compatibility
    
    app = QApplication(sys.argv)
    
    # Modern dark theme styling
    app.setStyle("Fusion")
    
    window = AutoAnnotateApp(video_path=args.video, model_path=args.model)
    window.show()
    sys.exit(app.exec())
