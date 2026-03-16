import os
import shutil
import base64
import random
import requests
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

# Configuration limits
# Base directory is the same directory as this script (the Train folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "extracted_frames")
YES_DIR = os.path.join(BASE_DIR, "import_to_label_studio")
NO_DIR = os.path.join(BASE_DIR, "background_frames")
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Prompts
SYSTEM_PROMPT =  """You are an expert automated data curation assistant analyzing Apex Legends gameplay footage. Your task is to determine if the player is actively engaged in a fight. You must output exactly one word: 'YES' if active combat is occurring, or 'NO' if the player is looting, rotating, in the lobby, or not actively fighting. Do not provide explanations, context, or additional text."""

USER_PROMPT =  """Analyze this Apex Legends gameplay frame. Is the player currently in an active fight? Focus heavily on the center of the screen and look for ANY of the following specific Apex combat indicators:

Floating damage numbers (colored White, Blue, Purple, Red, or Gold) appearing near the crosshair.

A "broken shield" shatter icon appearing next to the crosshair.

Large text reading "KNOCKED DOWN" or "ELIMINATED" appearing prominently in the upper-middle of the screen.

Red curved arrows (directional damage indicators) forming a ring around the center of the screen, showing the player is taking hits.

A distinct red 'X' hit marker flashing on the center crosshair.

If ANY of these specific UI elements are clearly visible, output YES. Otherwise, output NO."""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(img_name):
    img_path = os.path.join(INPUT_DIR, img_name)
    base64_image = encode_image(img_path)

    payload = {
        "model": "local-model", # LM Studio uses whichever model is currently loaded
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        answer = result['choices'][0]['message']['content'].strip().upper()
        
        if "YES" in answer:
            shutil.move(img_path, os.path.join(YES_DIR, img_name))
            return True, "YES"
        elif "NO" in answer:
            # Keep a random 10% of NO frames for background/negative examples
            if random.random() < 0.10:
                shutil.move(img_path, os.path.join(NO_DIR, img_name))
                return True, "NO (Saved sample)"
            else:
                os.remove(img_path)
                return True, "NO (Discarded)"
        else:
            return False, f"UNEXPECTED RESPONSE: {answer}"
            
    except Exception as e:
        return False, f"Error processing {img_name}: {e}"

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please create it and run 01_extract_frames.py to populate it.")
        return

    os.makedirs(YES_DIR, exist_ok=True)
    os.makedirs(NO_DIR, exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in os.listdir(INPUT_DIR) if Path(f).suffix.lower() in valid_extensions]

    print(f"Found {len(image_files)} images in {INPUT_DIR}.")
    if not image_files:
        return

    print("Starting processing with 6 concurrent workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_image, img): img for img in image_files}
        
        # Use tqdm for the progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtering Frames"):
            img_name = futures[future]
            try:
                success, msg = future.result()
                if not success:
                    tqdm.write(f"[{img_name}] {msg}")
            except Exception as exc:
                tqdm.write(f"[{img_name}] generated an exception: {exc}")

if __name__ == "__main__":
    main()
