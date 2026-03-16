from ultralytics import YOLO
import argparse
import os

def train_yolo(data_yaml, epochs=300, batch=16, imgsz=1088, model_weights="yolov8l.pt"):
    if not os.path.exists(data_yaml):
        print(f"Error: dataset config file {data_yaml} not found.")
        return
        
    print(f"Loading base model: {model_weights}")
    model = YOLO(model_weights)
    
    # Explicitly set the base directory for runs
    train_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(train_dir, "runs", "detect")

    print(f"Starting YOLO Training with data={data_yaml}, epochs={epochs}, batch={batch}, imgsz={imgsz}...")
    # Let Ultralytics auto-detect the best device (cuda if available, else cpu)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project_dir,
        patience=50,
        close_mosaic=15,
        
        # General Gaming Optimizations
        optimizer='AdamW',
        lr0=0.001,
        
        # Color Augmentations (helps separate UI from background)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Spatial Augmentations
        translate=0.1,
        scale=0.5,
        
        # Disable Flipping (NEVER flip gaming UI because text becomes backwards!)
        fliplr=0.0,
        
        # Windows Stability Fix
        workers=0
    )
    
    print("Training finished!")
    print("Check out the 'runs/detect/train' folder for your custom model weights and validation results.")

if __name__ == "__main__":
    train_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(train_dir, "YOLO_Data", "dataset.yaml")
    
    parser = argparse.ArgumentParser("Train YOLO custom model")
    parser.add_argument("--data", type=str, default=default_data_path, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1088, help="Image size")
    parser.add_argument("--base", type=str, default="yolov8x.pt", help="Base model weights")
    args = parser.parse_args()
    
    train_yolo(args.data, args.epochs, args.batch, args.imgsz, args.base)
