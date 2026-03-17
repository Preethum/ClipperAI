import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT (.engine) format for massive inference speedups.")
    parser.add_argument("--model", type=str, required=True, help="Path to your trained .pt model file (e.g., models/apex_gaming.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference (default 640 for YOLO)")
    parser.add_argument("--half", action="store_true", help="Enable FP16 half-precision in export (speeds up inference on supported GPUs)")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    
    print(f"Exporting to TensorRT (.engine)... This may take 5-10 minutes.")
    try:
        # Export the model
        exported_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=args.half,
            dynamic=True,
            verbose=True
        )
        print(f"\n✅ Export successful! Model saved to: {exported_path}")
        print("\nTo use it in the gaming pipeline, simply pass the .engine file into the --model parameter:")
        print(f"  python src/scenarios/gaming_pipeline.py --model models/apex_gaming.engine ...")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("💡 Note: TensorRT requires an NVIDIA GPU with TensorRT and CUDA drivers installed.")

if __name__ == "__main__":
    main()
