import os
import sys
import glob
import shutil
import random
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed. Please install it using 'py -m pip install ultralytics'")
    sys.exit(1)

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "DataSets", "provide_more")
SPLIT_DIR = os.path.join(BASE_DIR, "DataSets", "provide_more_split")

CATEGORIES = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Verify source directories
if not os.path.exists(SRC_DIR):
    print(f"ERROR: Source directory '{SRC_DIR}' does not exist.")
    sys.exit(1)

# Check if there are any images at all
total_images = 0
for cat in CATEGORIES:
    cat_path = os.path.join(SRC_DIR, cat)
    if os.path.exists(cat_path):
        total_images += len(glob.glob(os.path.join(cat_path, "*.*")))

if total_images == 0:
    print("ERROR: No images found inside 'DataSets/provide_more' subfolders.")
    print("Please run 'download_and_curate.py' first to download and prepare the dataset.")
    sys.exit(1)

print(f"Found {total_images} total images across 7 classes in '{SRC_DIR}'.")

# ==========================================
# 2. AUTOMATED DATASET AUDITING & SPLIT
# ==========================================
print("\n==========================================")
print("Auditing Images & Creating Train/Val Split (80/20)...")
print("==========================================")

# Clean up previous split directory if exists to avoid stale accumulation
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)

# Create train/val structure
for split in ['train', 'val']:
    for cat in CATEGORIES:
        os.makedirs(os.path.join(SPLIT_DIR, split, cat), exist_ok=True)

random.seed(42)  # For reproducible splits

corrupted_count = 0
copied_counts = {'train': 0, 'val': 0}

for cat in CATEGORIES:
    cat_src_path = os.path.join(SRC_DIR, cat)
    if not os.path.exists(cat_src_path):
        continue
        
    # Get all potential image files
    all_files = []
    for ext in ['*.*']:
        all_files.extend(glob.glob(os.path.join(cat_src_path, ext)))
        
    # Filter only valid image extensions
    valid_images = []
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp']:
            valid_images.append(f)
            
    # Shuffle and split
    random.shuffle(valid_images)
    
    # Audit image integrity
    clean_images = []
    for f in valid_images:
        try:
            with Image.open(f) as img:
                img.verify()  # Fast verification of structural integrity
            clean_images.append(f)
        except Exception:
            corrupted_count += 1
            print(f"  [Curation] Removed corrupted image: {os.path.basename(f)}")
            
    split_idx = int(len(clean_images) * 0.8)
    train_files = clean_images[:split_idx]
    val_files = clean_images[split_idx:]
    
    # Copy files
    for f in train_files:
        shutil.copy2(f, os.path.join(SPLIT_DIR, 'train', cat, os.path.basename(f)))
        copied_counts['train'] += 1
    for f in val_files:
        shutil.copy2(f, os.path.join(SPLIT_DIR, 'val', cat, os.path.basename(f)))
        copied_counts['val'] += 1

print(f"Dataset preparation complete:")
print(f"  - Valid images copied to train: {copied_counts['train']}")
print(f"  - Valid images copied to val: {copied_counts['val']}")
if corrupted_count > 0:
    print(f"  - Corrupted files skipped: {corrupted_count} (successfully filtered to avoid GPU crashes)")

# ==========================================
# 3. GPU CUDA HARDWARE CHECK
# ==========================================
print("\n==========================================")
print("Hardware Check for GPU Acceleration...")
print("==========================================")
device_choice = 'cpu'
if torch.cuda.is_available():
    device_choice = 0
    print(f"SUCCESS: NVIDIA GPU detected with CUDA support!")
    print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  - Device index: 0 (will be used for accelerated training)")
else:
    print("WARNING: No NVIDIA GPU detected or CUDA is not configured properly.")
    print("  - Training will fall back to CPU. This may take longer to complete.")

# ==========================================
# 4. TRAINING YOLO MODEL
# ==========================================
print("\n==========================================")
print("Initiating YOLO26 Model Training on GPU/CPU...")
print("==========================================")

base_model_path = os.path.join(BASE_DIR, "yolo26n-cls.pt")
if not os.path.exists(base_model_path):
    print(f"Base model '{base_model_path}' not found, Ultralytics will auto-download yolo26n-cls.pt.")

try:
    # Load base pre-trained model
    model = YOLO(base_model_path if os.path.exists(base_model_path) else 'yolo26n-cls.pt')
    
    # Train classification model
    # epochs=10: rapid GPU fine-tuning epochs
    # imgsz=224: standard classification size
    results = model.train(
        data=SPLIT_DIR,
        epochs=10,
        imgsz=224,
        device=device_choice,
        workers=2,
        project="yolo_waste_runs",
        name="train_cls"
    )
    
    # Locate best.pt weight file
    best_weight_path = os.path.join(BASE_DIR, "yolo_waste_runs", "train_cls", "weights", "best.pt")
    target_weight_path = os.path.join(BASE_DIR, "yolo26_waste.pt")
    
    if os.path.exists(best_weight_path):
        shutil.copy2(best_weight_path, target_weight_path)
        print("\n==========================================")
        print("CONGRATULATIONS: Training Completed Successfully!")
        print(f"  - Best weights successfully saved as: {target_weight_path}")
        print("  - Restart app.py to immediately activate your custom GPU-trained YOLO26 model!")
        print("==========================================")
    else:
        print(f"ERROR: Could not locate best weights file at: {best_weight_path}")
        
except Exception as train_err:
    print(f"\nFATAL: Training session encountered an error: {train_err}")
    sys.exit(1)
