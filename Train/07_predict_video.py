import sys
import os
import argparse
import time

# IMPORTANT DECORATOR FIX FOR WINDOWS [WinError 1114]
# Load PyTorch/Ultralytics before cv2 to prevent massive C++ DLL initialization clashes
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def process_video(video_path, model_path, output_path, conf_threshold):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    print(f"Loading YOLO model from {os.path.basename(model_path)}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open input video.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object
    # mp4v is the standard format for generating .mp4 files with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video writer at {output_path}")
        cap.release()
        return

    print(f"\n🎬 Processing Video: {os.path.basename(video_path)}")
    print(f"💾 Output Destination: {output_path}")
    print(f"📏 Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"🎞️ Total Frames: {total_frames}\n")

    start_time = time.time()
    
    for _ in tqdm(range(total_frames), desc="Annotating Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference on the clean frame without printing stdout logs for every frame
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Plot the bounding boxes onto the frame
        # line_width=2 and font_size=1 to avoid extremely unreadable giant text blocks
        annotated_frame = results[0].plot(line_width=2, font_size=1) 
        
        # Write the painted frame to the new video file
        out.write(annotated_frame)

    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Video processing complete in {elapsed:.1f} seconds!")
    print(f"🎉 Fully annotated video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an entire video with a trained YOLO model and output an annotated video.")
    parser.add_argument("--video", required=True, help="Path to the input raw video file (.mp4)")
    parser.add_argument("--model", required=True, help="Path to your trained YOLO model (.pt)")
    parser.add_argument("--output", default="annotated_output.mp4", help="Path for the newly generated output video file (default: annotated_output.mp4)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for bounding boxes (default: 0.5)")
    args = parser.parse_args()

    # Ensure output directory tree exists if a nested path is provided (e.g., 'outputs/final_video.mp4')
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    process_video(args.video, args.model, args.output, args.conf)
