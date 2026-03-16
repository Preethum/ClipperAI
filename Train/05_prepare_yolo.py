import os
import shutil
import random
import glob

def prepare_dataset(input_dir, output_dir, split_ratio=0.8):
    """
    Shuffles data and splits it into train and validation sets for YOLO.
    Assumes your input_dir contains the text files and images from Label Studio YOLO export.
    """
    print(f"Preparing YOLO dataset from {input_dir} into {output_dir}")
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Export directory needs 'images/' and 'labels/' subfolders from Label Studio.")
        return

    # Find all images
    valid_extensions = ('.jpg', '.png', '.jpeg')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    # Shuffle
    random.shuffle(image_files)
    
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def move_files(files, split_name):
        for file in files:
            # Copy image
            src_img = os.path.join(images_dir, file)
            dst_img = os.path.join(output_dir, 'images', split_name, file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_file = os.path.splitext(file)[0] + '.txt'
            src_lbl = os.path.join(labels_dir, label_file)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(output_dir, 'labels', split_name, label_file)
                shutil.copy2(src_lbl, dst_lbl)

    move_files(train_files, 'train')
    move_files(val_files, 'val')
    print(f"Copied {len(train_files)} to train, {len(val_files)} to val.")

    # Parse classes.txt to dynamically build the YAML map
    classes_file = os.path.join(input_dir, 'classes.txt')
    names_str = ""
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        for idx, line in enumerate(lines):
            names_str += f"  {idx}: '{line}'\n"
    else:
        names_str = "  0: target_indicator\n"

    # Create dataset.yaml with explicit absolute paths
    abs_output_dir = os.path.abspath(output_dir).replace('\\', '/')
    yaml_content = f"""path: {abs_output_dir}
train: images/train
val: images/val

names:
{names_str}"""
    with open(os.path.join(input_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"Created dataset.yaml in {input_dir}")

if __name__ == "__main__":
    import argparse
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "YOLO_Data"), help="Path to unzipped label studio export")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "yolo_dataset"), help="Output directory")
    args = parser.parse_args()
    
    # Check if the directories actually exist instead of silently failing
    images_dir = os.path.join(args.input, 'images')
    labels_dir = os.path.join(args.input, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"\n❌ ERROR: Could not find 'images/' or 'labels/' inside the directory: {args.input}")
        print("Please export your Label Studio project in YOLO format, unzip it, and place the contents into the YOLO_Data folder.\n")
        exit(1)
        
    prepare_dataset(args.input, args.output)
