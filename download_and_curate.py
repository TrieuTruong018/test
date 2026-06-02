import os
import sys
import glob
import shutil
import cv2
import yaml
from PIL import Image
from roboflow import Roboflow

# ==========================================
# 1. ROBOFLOW API KEY & DATASETS DEFINITION
# ==========================================
ROBOFLOW_API_KEY = "zndtLcJJQp3NHm5MPKpM"

if ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY" or not ROBOFLOW_API_KEY:
    # Attempt to read from environment or prompt
    ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
    if not ROBOFLOW_API_KEY:
        try:
            ROBOFLOW_API_KEY = input("Enter your Roboflow API Key: ").strip()
        except KeyboardInterrupt:
            sys.exit(1)

if not ROBOFLOW_API_KEY:
    print("ERROR: Roboflow API key is required to download the datasets.")
    sys.exit(1)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

datasets = [
    {
        "name": "waste-classification-final",
        "workspace": "class-work-ggaxv",
        "project": "waste-classification-final",
        "version": 1,
        "location": "./datasets/waste-classification-final",
    },
    {
        "name": "compost-or-not-compost",
        "workspace": "capstone-upocg",
        "project": "compost-or-not-compost",
        "version": 2,
        "location": "./datasets/compost-or-not-compost",
    },
    {
        "name": "grabage-detection",
        "workspace": "waste-classifiaction-system",
        "project": "grabage-detection",
        "version": 1,
        "location": "./datasets/grabage-detection",
    },
]

# Target 7 categories directory
TARGET_DIR = os.path.join("DataSets", "provide_more")
os.makedirs(TARGET_DIR, exist_ok=True)

CATEGORIES = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
for cat in CATEGORIES:
    os.makedirs(os.path.join(TARGET_DIR, cat), exist_ok=True)

# ==========================================
# 2. CLASS MAPPING LOGIC (CASE-INSENSITIVE KEYWORDS)
# ==========================================
def map_class_name(name):
    """
    Maps dynamic class names from Roboflow crowdsourced datasets into our 7 target categories.
    """
    name = name.lower().strip()
    
    # Glass
    if any(k in name for k in ['glass', 'bottle_glass', 'glass_bottle', 'thuy tinh', 'thuy_tinh']):
        return 'glass'
    # Plastic
    if any(k in name for k in ['plastic', 'bottle_plastic', 'plastic_bottle', 'pet', 'nilon', 'nhua']):
        return 'plastic'
    # Cardboard
    if any(k in name for k in ['cardboard', 'carton', 'box', 'paper_box', 'bia', 'bia_carton']):
        return 'cardboard'
    # Paper
    if any(k in name for k in ['paper', 'newspaper', 'book', 'magazine', 'office_paper', 'giay']):
        return 'paper'
    # Metal
    if any(k in name for k in ['metal', 'can', 'aluminum', 'tin', 'steel', 'kim loai', 'kim_loai', 'lon', 'lon_nhom']):
        return 'metal'
    # Compostable
    if any(k in name for k in ['compost', 'food', 'organic', 'vegetable', 'fruit', 'biodegradable', 'leaf', 'leaves', 'huu co', 'huu_co', 'rau', 'trai_cay']):
        return 'compost'
    # General Trash / Landfill
    if any(k in name for k in ['trash', 'garbage', 'refuse', 'non-recyclable', 'landfill', 'waste', 'rac', 'rac_thai']):
        return 'trash'
        
    return None

# ==========================================
# 3. DOWNLOAD DATASETS FROM ROBOFLOW
# ==========================================
for ds in datasets:
    print(f"\n==========================================")
    print(f"Downloading {ds['name']}...")
    print(f"==========================================")
    try:
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        version = project.version(ds["version"])
        dataset = version.download(
            model_format="yolov8",
            location=ds["location"],
            overwrite=True,
        )
        print(f"Downloaded successfully to: {dataset.location}")
    except Exception as e:
        print(f"ERROR downloading {ds['name']}: {e}")
        continue

# ==========================================
# 4. DATA CURATION & AUTO-PLOTTING CROPS
# ==========================================
print("\n==========================================")
print("Starting Data Curation & Crop Extraction...")
print("==========================================")

crop_counts = {cat: 0 for cat in CATEGORIES}
unmapped_classes = set()

for ds in datasets:
    ds_path = ds["location"]
    if not os.path.exists(ds_path):
        continue
        
    yaml_files = glob.glob(os.path.join(ds_path, "*.yaml"))
    if not yaml_files:
        print(f"No YAML config found for {ds['name']}, skipping curation.")
        continue
        
    # Read Class names from data.yaml
    with open(yaml_files[0], 'r', encoding='utf-8') as f:
        try:
            yaml_data = yaml.safe_load(f)
            class_names = yaml_data.get('names', {})
            if isinstance(class_names, list):
                class_names = {idx: name for idx, name in enumerate(class_names)}
        except Exception as e:
            print(f"Error reading YAML for {ds['name']}: {e}")
            continue
            
    print(f"Processing dataset '{ds['name']}' with categories: {list(class_names.values())}")
    
    # Build local class mapper
    local_class_map = {}
    for idx, name in class_names.items():
        mapped_cat = map_class_name(name)
        if mapped_cat:
            local_class_map[idx] = mapped_cat
            print(f"  Mapped class '{name}' -> '{mapped_cat}'")
        else:
            unmapped_classes.add(name)
            print(f"  [Skipped] Could not map class '{name}'")
            
    # Search all images inside the dataset recursively
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(ds_path, "**", ext), recursive=True))
        
    print(f"Found {len(image_paths)} images in {ds['name']}. Extrapolating object crops...")
    
    for img_path in image_paths:
        # Check corresponding YOLO labels file
        img_dir, img_filename = os.path.split(img_path)
        img_name, _ = os.path.splitext(img_filename)
        
        # Labels are usually in a sibling "labels" directory
        label_dir = img_dir.replace("images", "labels")
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        
        # Load image once using OpenCV with PIL fallback for Unicode support
        img = None
        try:
            img = cv2.imread(img_path)
        except Exception:
            pass
            
        if img is None:
            try:
                # PIL fallback
                pil_img = Image.open(img_path).convert('RGB')
                import numpy as np
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                # Corrupted file, skip to enforce dataset curation cleanliness
                continue
                
        h, w, _ = img.shape
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as lf:
                lines = lf.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_idx = int(parts[0])
                if class_idx not in local_class_map:
                    continue
                    
                target_cat = local_class_map[class_idx]
                
                # YOLO Normalized coordinates: x_center, y_center, width, height
                x_c, y_c, box_w, box_h = map(float, parts[1:5])
                
                # Convert back to pixel coordinates
                xmin = int((x_c - box_w / 2) * w)
                ymin = int((y_c - box_h / 2) * h)
                xmax = int((x_c + box_w / 2) * w)
                ymax = int((y_c + box_h / 2) * h)
                
                # Boundary clamping
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                
                # Skip invalid or ultra-small crops
                if xmax - xmin < 10 or ymax - ymin < 10:
                    continue
                    
                # Crop and save!
                crop = img[ymin:ymax, xmin:xmax]
                crop_filename = f"{ds['name']}_{img_name}_crop{idx}.jpg"
                crop_save_path = os.path.join(TARGET_DIR, target_cat, crop_filename)
                
                # Encode and write
                cv2.imwrite(crop_save_path, crop)
                crop_counts[target_cat] += 1
        else:
            # Classification fallback / Single image mode: If no box annotations, check if class name is in folder path
            # Look up path keywords to guess class
            path_lower = img_path.lower()
            guessed_cat = None
            for idx, name in class_names.items():
                if name.lower() in path_lower and idx in local_class_map:
                    guessed_cat = local_class_map[idx]
                    break
                    
            if guessed_cat:
                save_filename = f"{ds['name']}_{img_filename}"
                save_path = os.path.join(TARGET_DIR, guessed_cat, save_filename)
                cv2.imwrite(save_path, img)
                crop_counts[guessed_cat] += 1

print("\n==========================================")
print("Data Curation Summary (Images extracted into provide_more):")
print("==========================================")
for cat, count in crop_counts.items():
    print(f"  - {cat.upper()}: {count} images")
if unmapped_classes:
    print(f"Unmapped class keywords: {list(unmapped_classes)}")
print("\nSuccess! Dataset is beautifully curated. Ready for training.")
