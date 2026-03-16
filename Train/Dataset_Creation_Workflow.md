# Dataset Creation Workflow

This workflow has been broken down into executable scripts for each step. Run these scripts in order to build your YOLO dataset from raw gameplay videos.

## Step 1: Frame Extraction
**Script:** `01_extract_frames.py`

This script takes your raw gameplay video and breaks it into individual images using FFmpeg at a set interval (default 1 frame per second).

**How to run:**
```cmd
python 01_extract_frames.py --video path/to/your/gameplay.mp4 --out extracted_frames --fps 1
```

## Step 2: Vision LLM Filtering Configuration
**File:** `02_lm_studio_prompts.md`

This file contains the strict System Prompt and User Prompt logic needed to lock down the LM Studio behavior. 
1. Open LM Studio and start the Local Server on `http://localhost:1234`.
2. Load your desired Vision LLM.

## Step 3: Automated Sorting
**Script:** `03_filter_frames.py`

This script loops through your `extracted_frames` folder, sends the images to your local LM Studio API using the prompts from Step 2, and sorts the files. "YES" frames go to `import_to_label_studio` and "NO" frames go to `background_frames` (keeping a random 10% sample).

**How to run:**
```cmd
python 03_filter_frames.py
```
*(Make sure to edit `USER_PROMPT` in the script to include your specific indicator before running!)*

## Step 3b: Pre-Labeling (Optional but Recommended)
**Script:** `03b_auto_annotate.py`

This script uses an existing Ultralytics YOLO model (like the base `yolov8n.pt` or a previously trained model) to auto-generate bounding boxes (`.txt` files) for all the images in your `import_to_label_studio` folder. By running this, you can import pre-annotated data into Label Studio, meaning you only have to correct boxes rather than draw them from scratch!

**How to run:**
```cmd
python 03b_auto_annotate.py --model yolov8n.pt --conf 0.25
```

## Step 4: Label Studio Annotation
**Script:** `04_launch_label_studio.bat`

This batch file launches Label Studio with local file serving on port 8080. 
Create a local storage connection in Label Studio pointing to your `import_to_label_studio` folder. Keep in mind you need to install label-studio (`pip install label-studio`) first.

**How to run:**
Double-click `04_launch_label_studio.bat` or run:
```cmd
./04_launch_label_studio.bat
```

## Step 5: Dataset Preparation
**Script:** `05_prepare_yolo.py`

Once you have exported your annotations from Label Studio in the YOLO format, run this script. It shuffles your data, splits it into 80/20 train and validation sets, and generates the `dataset.yaml` file.

**How to run:**
```cmd
python 05_prepare_yolo.py --input path/to/label_studio_export_folder --output yolo_dataset
```

## Step 6: YOLO Training
**Script:** `06_train_yolo.py`

This script points the Ultralytics framework at your new `dataset.yaml` file and starts the training loop to produce a highly accurate custom `.pt` model weight file.

**How to run:**
```cmd
python 06_train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```
