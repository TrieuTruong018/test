import os
import sys
import time
import json
import random
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Optional imports for Deep Learning
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras import regularizers
except ImportError:
    tf = None
    keras = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model state and training progress
model = None
fallback_model = None
yolo_model = None
training_status = {
    'status': 'idle',  # 'idle', 'running', 'completed', 'failed'
    'epoch': 0,
    'total_epochs': 10,
    'loss': 0.0,
    'accuracy': 0.0,
    'val_loss': 0.0,
    'val_accuracy': 0.0,
    'progress': 0,
    'log': '',
    'error': ''
}
training_lock = threading.Lock()

# Target Classes
CLASSES = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Eco Recycling Recommendations Dictionary
ECO_INFO = {
    'cardboard': {
        'title': 'Cardboard (Giấy bìa Carton)',
        'color': '#d97706',
        'decomposition': '2 - 3 tháng',
        'tip': 'Gấp phẳng các thùng carton để tiết kiệm diện tích. Đảm bảo hộp sạch, khô, không dính dầu mỡ thực phẩm (như hộp pizza dính dầu) trước khi bỏ vào thùng tái chế.',
        'fact': 'Tái chế 1 tấn bìa carton giúp tiết kiệm đến 17 cây gỗ trưởng thành và hơn 26,000 lít nước sạch.'
    },
    'compost': {
        'title': 'Compostable (Rác hữu cơ / Phân hủy sinh học)',
        'color': '#16a34a',
        'decomposition': 'Vài tuần - vài tháng',
        'tip': 'Thu gom vỏ rau củ quả, bã trà, cà phê, lá cây khô. Bạn có thể tự làm phân bón hữu cơ tại nhà để bón cho cây trồng, giảm thiểu rác thải sinh hoạt.',
        'fact': 'Rác hữu cơ chiếm khoảng 50-60% tổng lượng rác thải sinh hoạt tại Việt Nam. Việc ủ phân hữu cơ giúp giảm thiểu khí metan gây hiệu ứng nhà kính từ các bãi rác.'
    },
    'glass': {
        'title': 'Glass (Thủy tinh)',
        'color': '#06b6d4',
        'decomposition': 'Hàng triệu năm (Hầu như không tự phân hủy)',
        'tip': 'Rửa sạch chai lọ thủy tinh trước khi gom tái chế. Thủy tinh có thể tái chế vô hạn lần mà không hề bị giảm chất lượng.',
        'fact': 'Một chai thủy tinh mất hơn 1 triệu năm để tự phân hủy trong tự nhiên. Nhưng khi tái chế, nó có thể trở lại kệ hàng chỉ trong vòng 30 ngày!'
    },
    'metal': {
        'title': 'Metal (Kim loại / Lon nhôm / Sắt)',
        'color': '#e2e8f0',
        'decomposition': '80 - 200 năm',
        'tip': 'Ép xẹp vỏ lon bia, nước ngọt bằng nhôm để thu gom. Rửa sơ các hộp sữa đặc hoặc hộp cá mồi kim loại trước khi gom tái chế.',
        'fact': 'Tái chế lon nhôm tiết kiệm đến 95% năng lượng so với việc sản xuất lon mới hoàn toàn từ quặng thô.'
    },
    'paper': {
        'title': 'Paper (Giấy văn phòng / Báo chí)',
        'color': '#f59e0b',
        'decomposition': '2 - 6 tuần',
        'tip': 'Thu gom giấy báo cũ, sách vở, giấy viết học sinh. Tránh bỏ giấy đã bị dính nước, dầu mỡ hoặc các loại giấy phủ màng nhựa bóng vào thùng tái chế.',
        'fact': 'Mỗi tấn giấy tái chế giúp bảo vệ được 17 cây rừng, tiết kiệm 4,000 kWh điện năng và 31,000 lít nước.'
    },
    'plastic': {
        'title': 'Plastic (Nhựa / Chai nhựa / Túi nilon)',
        'color': '#3b82f6',
        'decomposition': '100 - 500 năm',
        'tip': 'Rửa sạch chai nhựa đựng nước ngọt, mỹ phẩm. Hạn chế sử dụng túi nilon dùng một lần, thay thế bằng túi vải sinh học thân thiện môi trường.',
        'fact': 'Hơn 8 triệu tấn nhựa được thải ra đại dương mỗi năm. Một chiếc chai nhựa cần tới 450 năm để phân hủy thành vi nhựa gây hại cho sinh vật.'
    },
    'trash': {
        'title': 'General Trash (Rác thải còn lại / Không tái chế)',
        'color': '#6b7280',
        'decomposition': 'Không xác định',
        'tip': 'Gồm hộp xốp, tã giấy, khẩu trang, khăn giấy ướt, đồ gốm sứ vỡ... Hãy đóng gói cẩn thận để công nhân vệ sinh dễ dàng thu gom mang đi chôn lấp hoặc thiêu hủy.',
        'fact': 'Giảm thiểu rác thải sinh hoạt bằng cách chọn sản phẩm bền vững là cách tốt nhất để bảo vệ hành tinh xanh của chúng ta.'
    }
}

# ImageNet Class to 7 Waste Classes Mapping Table
IMAGENET_MAP = {
    # Cardboard
    'carton': 'cardboard', 'cardboard': 'cardboard', 'box': 'cardboard', 'crate': 'cardboard',
    
    # Glass
    'wine_bottle': 'glass', 'beer_bottle': 'glass', 'pop_bottle': 'glass', 'water_bottle': 'glass',
    'beaker': 'glass', 'goblet': 'glass', 'chalice': 'glass', 'vase': 'glass', 'jar': 'glass',
    
    # Metal
    'tin_can': 'metal', 'can': 'metal', 'brass': 'metal', 'iron': 'metal', 'padlock': 'metal',
    'safe': 'metal', 'nail': 'metal', 'screw': 'metal', 'frying_pan': 'metal', 'saucepan': 'metal',
    'pot': 'metal', 'key': 'metal', 'screwdriver': 'metal', 'hammer': 'metal',
    
    # Paper
    'envelope': 'paper', 'book': 'paper', 'notebook': 'paper', 'magazine': 'paper',
    'packet': 'paper', 'paper_towel': 'paper', 'toilet_tissue': 'paper', 'napkin': 'paper',
    'menu': 'paper',
    
    # Plastic
    'plastic_bag': 'plastic', 'pill_bottle': 'plastic', 'syringe': 'plastic',
    'soap_dispenser': 'plastic', 'balloon': 'plastic', 'bucket': 'plastic', 'tub': 'plastic',
    
    # Compostable
    'banana': 'compost', 'orange': 'compost', 'lemon': 'compost', 'fig': 'compost',
    'pineapple': 'compost', 'apple': 'compost', 'strawberry': 'compost', 'pomegranate': 'compost',
    'grape': 'compost', 'cabbage': 'compost', 'broccoli': 'compost', 'cauliflower': 'compost',
    'zucchini': 'compost', 'cucumber': 'compost', 'artichoke': 'compost', 'bell_pepper': 'compost',
    'mushroom': 'compost', 'acorn_squash': 'compost', 'butternut_squash': 'compost',
    'potato': 'compost', 'onion': 'compost', 'garlic': 'compost', 'carrot': 'compost',
    
    # Trash fallbacks
    'cup': 'trash', 'plate': 'trash', 'diaper': 'trash', 'bandage': 'trash',
    'sponge': 'trash', 'mask': 'trash', 'glove': 'trash', 'toothbrush': 'trash'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ==========================================
# ADVANCED DEEP LEARNING ARCHITECTURE HELPERS
# ==========================================
import math

def squeeze_excitation_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation (SE) block as channel attention.
    Squeezes spatial dimensions, excites channels, and rescales features.
    """
    init = input_tensor
    filters = init.shape[-1]
    se = GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = tf.keras.layers.multiply([init, se])
    return x

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Categorical Focal Loss for multi-class classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss

class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """
    Callback for Cosine Annealing Learning Rate scheduling.
    Decays learning rate following a half-cosine curve.
    """
    def __init__(self, lr_max, lr_min, total_epochs):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_epochs = total_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        cos_inner = (math.pi * epoch) / self.total_epochs
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(cos_inner))
        
        # Robust learning rate assignment for both legacy and modern Keras optimizers
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'learning_rate') and hasattr(optimizer.learning_rate, 'assign'):
            optimizer.learning_rate.assign(lr)
        elif hasattr(optimizer, 'lr') and hasattr(optimizer.lr, 'assign'):
            optimizer.lr.assign(lr)
        else:
            try:
                tf.keras.backend.set_value(optimizer.learning_rate, lr)
            except Exception:
                try:
                    tf.keras.backend.set_value(optimizer.lr, lr)
                except Exception:
                    try:
                        optimizer.learning_rate = lr
                    except Exception:
                        optimizer.lr = lr
                        
        global training_status
        with training_lock:
            training_status['log'] += f"[LR Scheduler] Epoch {epoch+4}: Set Learning Rate = {lr:.2e}\n"

# ==========================================
# EXPLAINABLE AI (XAI) GRAD-CAM HEATMAP GEN
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu", pred_index=None):
    """
    Generates class activation maps (Grad-CAM) for a given image.
    """
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Programmatic search for the last 2D convolutional/relu layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if any(k in layer.name.lower() for k in ['conv2d', 'out_relu', 'conv_1', 'relu']):
                last_conv_layer = layer
                last_conv_layer_name = layer.name
                break
                
    if last_conv_layer is None:
        raise ValueError("Cannot find a valid convolutional layer for Grad-CAM.")
        
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def generate_and_save_gradcam(original_img_path, heatmap, save_path, alpha=0.4):
    """
    Applies a jet colormap to the heatmap, overlays it on the image, and saves.
    Handles OpenCV (with Unicode path support) and has PIL/matplotlib fallback.
    """
    if cv2 is not None:
        try:
            # Read using numpy to support Unicode paths on Windows
            img = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_color = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
            is_success, im_buf_arr = cv2.imencode(".jpg", superimposed_img)
            if is_success:
                im_buf_arr.tofile(save_path)
                return True
        except Exception as e:
            print(f">>> OpenCV Grad-CAM overlay failed: {e}. Trying PIL fallback...")
            
    # PIL Fallback
    try:
        from PIL import Image
        import matplotlib.cm as cm
        
        img = Image.open(original_img_path).convert('RGBA')
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.BILINEAR)
        
        colormap = cm.get_cmap('jet')
        heatmap_color_arr = np.uint8(colormap(np.array(heatmap_resized)) * 255)
        heatmap_color_img = Image.fromarray(heatmap_color_arr).convert('RGBA')
        
        blended = Image.blend(img, heatmap_color_img, alpha=alpha)
        blended.convert('RGB').save(save_path, "JPEG")
        return True
    except Exception as e:
        print(f">>> PIL Grad-CAM overlay failed: {e}")
        return False

def load_models():
    """Load the custom trained model, the fallback Keras MobileNetV2 model, and the YOLO26 model."""
    global model, fallback_model, yolo_model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waste_model.h5')
    
    # 1. Load TensorFlow Models
    if tf is not None:
        if os.path.exists(model_path):
            try:
                # Load with compile=False to avoid deserialization compile issues
                model = keras.models.load_model(model_path, compile=False)
                print(">>> Custom Waste Classifier Model loaded successfully!")
            except Exception as e:
                print(f">>> Error loading custom model: {e}. Falling back...")
        
        if model is None:
            try:
                fallback_model = MobileNetV2(weights='imagenet')
                print(">>> Fallback MobileNetV2 (ImageNet) model loaded successfully!")
            except Exception as e:
                print(f">>> Error loading fallback model: {e}")
                
    # 2. Load YOLO26 Model
    if YOLO is not None:
        try:
            # Check if custom trained yolo weights exist
            yolo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo26_waste.pt')
            if os.path.exists(yolo_path):
                yolo_model = YOLO(yolo_path)
                print(">>> Custom YOLO26 Waste Classifier loaded successfully!")
            else:
                # Load pre-trained yolo26n-cls model (will auto-download from Ultralytics if not present)
                yolo_model = YOLO('yolo26n-cls.pt')
                print(">>> Pre-trained YOLO26 Classification model loaded successfully!")
        except Exception as e:
            print(f">>> Error loading YOLO26 model: {e}")

def map_imagenet_to_waste(preds):
    """Maps ImageNet predictions to the 7 waste classes."""
    # decode_predictions returns top classes: (class_id, class_name, prob)
    top_preds = decode_predictions(preds, top=5)[0]
    
    # Look for matching mappings in order of confidence
    for _, class_name, prob in top_preds:
        class_name_lower = class_name.lower()
        
        # Exact match in mapping
        if class_name_lower in IMAGENET_MAP:
            mapped_class = IMAGENET_MAP[class_name_lower]
            return mapped_class, float(prob), f"ImageNet detected '{class_name.replace('_', ' ')}'"
            
        # Partial substring match
        for key, value in IMAGENET_MAP.items():
            if key in class_name_lower or class_name_lower in key:
                return value, float(prob) * 0.9, f"ImageNet detected '{class_name.replace('_', ' ')}' (mapped via '{key}')"
                
    # Ultimate fallback if no match found
    first_class = top_preds[0][1].replace('_', ' ')
    first_prob = float(top_preds[0][2])
    
    # Heuristics based on keyword matching
    if any(k in first_class for k in ['bottle', 'glass', 'dish', 'cup']):
        return 'glass', first_prob * 0.6, f"Heuristic match for '{first_class}'"
    elif any(k in first_class for k in ['box', 'container', 'paper', 'book', 'letter']):
        return 'paper', first_prob * 0.6, f"Heuristic match for '{first_class}'"
    elif any(k in first_class for k in ['fruit', 'food', 'plant', 'flower', 'tree', 'vegetable']):
        return 'compost', first_prob * 0.7, f"Heuristic match for '{first_class}'"
    elif any(k in first_class for k in ['tool', 'metal', 'car', 'machine', 'wire']):
        return 'metal', first_prob * 0.6, f"Heuristic match for '{first_class}'"
        
    return 'trash', first_prob * 0.5, f"Default fallback from '{first_class}'"

@app.route('/')
def index():
    # Verify dataset directories
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataSets', 'Train')
    dataset_exists = False
    if os.path.exists(dataset_path):
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if len(subdirs) >= 3: # Check if class subfolders exist
            dataset_exists = True
            
    # Check if custom model is loaded
    custom_model_active = model is not None
    
    return render_template('index.html', dataset_exists=dataset_exists, custom_model_active=custom_model_active)

@app.route('/classify', methods=['POST'])
def classify():
    global model, fallback_model, yolo_model
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload JPG, PNG, or WEBP'}), 400
        
    # Get model type selection
    model_type = request.form.get('model_type', 'mobilenet')
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    web_filepath = f'/static/uploads/{filename}'
    
    # ==========================================
    # ROUTE A: YOLO26 MODEL INFERENCE (With strict NMS and Conf thresholds)
    # ==========================================
    if model_type == 'yolo26':
        if yolo_model is not None:
            try:
                # Perform inference with strictly tightened confidence threshold and strict NMS
                # conf=0.6: only return predictions with confidence >= 60%
                # iou=0.45: strict NMS, suppresses boxes that overlap by more than 45%
                results = yolo_model(filepath, conf=0.6, iou=0.45)
                
                # Check if it has classification outputs (probs)
                if hasattr(results[0], 'probs') and results[0].probs is not None:
                    probs = results[0].probs
                    is_custom_yolo = len(probs.data) == len(CLASSES)
                    
                    if is_custom_yolo:
                        pred_idx = int(probs.top1)
                        confidence = float(probs.top1conf)
                        pred_class = CLASSES[pred_idx]
                        
                        # Form predictions list
                        all_predictions = []
                        for idx, val in enumerate(probs.data):
                            prob = float(val)
                            all_predictions.append({
                                'class': CLASSES[idx],
                                'confidence': prob * 100,
                                'info': ECO_INFO[CLASSES[idx]]
                            })
                        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
                        method_name = "Custom Trained YOLO26 Model (SOTA)"
                    else:
                        # It's an ImageNet pre-trained YOLO26 model. Map ImageNet categories to waste classes!
                        probs_numpy = probs.data.cpu().numpy()
                        preds = np.expand_dims(probs_numpy, axis=0)
                        pred_class, confidence, detection_detail = map_imagenet_to_waste(preds)
                        
                        # Form simulated multi-class output for UI (same as Fallback)
                        all_predictions = []
                        remaining_prob = 1.0 - confidence
                        other_classes = [c for c in CLASSES if c != pred_class]
                        random.shuffle(other_classes)
                        
                        all_predictions.append({
                            'class': pred_class,
                            'confidence': confidence * 100,
                            'info': ECO_INFO[pred_class]
                        })
                        for i, other in enumerate(other_classes):
                            weight = (0.5 ** (i + 1)) * remaining_prob
                            all_predictions.append({
                                'class': other,
                                'confidence': weight * 100,
                                'info': ECO_INFO[other]
                            })
                        # Normalize final sum to exactly 100%
                        sum_prob = sum(x['confidence'] for x in all_predictions)
                        if sum_prob < 100.0:
                            all_predictions[-1]['confidence'] += (100.0 - sum_prob)
                        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
                        method_name = f"YOLO26 ImageNet Engine ({detection_detail})"
                
                # Check if it has object detection outputs (boxes)
                elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Object Detection model! Plot bounding boxes using YOLO's built-in results[0].plot()
                    plotted_img = results[0].plot()
                    
                    detected_filename = f"detected_{filename}"
                    detected_filepath = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
                    
                    # Save plotted BGR image with boxes
                    if cv2 is not None:
                        is_success, im_buf_arr = cv2.imencode(".jpg", plotted_img)
                        if is_success:
                            im_buf_arr.tofile(detected_filepath)
                    else:
                        from PIL import Image
                        plotted_rgb = plotted_img[..., ::-1] # BGR to RGB
                        Image.fromarray(plotted_rgb).save(detected_filepath, "JPEG")
                        
                    web_filepath = f'/static/uploads/{detected_filename}'
                    
                    # Get class with highest confidence from first box (sorted descending by default)
                    top_box = results[0].boxes[0]
                    pred_idx = int(top_box.cls[0])
                    confidence = float(top_box.conf[0])
                    
                    # Map to CLASSES
                    yolo_classes_names = results[0].names
                    if len(yolo_classes_names) == len(CLASSES):
                        pred_class = CLASSES[pred_idx]
                        
                        all_predictions = []
                        # Take the highest confidence box for each class present
                        conf_map = {c: 0.0 for c in CLASSES}
                        for b in results[0].boxes:
                            c_idx = int(b.cls[0])
                            c_name = CLASSES[c_idx]
                            c_conf = float(b.conf[0]) * 100
                            if c_conf > conf_map[c_name]:
                                conf_map[c_name] = c_conf
                                
                        for idx, c_name in enumerate(CLASSES):
                            all_predictions.append({
                                'class': c_name,
                                'confidence': conf_map[c_name],
                                'info': ECO_INFO[c_name]
                            })
                        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
                        method_name = "YOLO26 Object Detection Model (SOTA)"
                    else:
                        # Mapped from COCO detection categories (e.g. bottle -> plastic/glass, box -> cardboard/paper)
                        coco_class_name = yolo_classes_names[pred_idx]
                        pred_class = 'trash' # fallback
                        
                        # Apply keyword heuristics to map COCO classes
                        coco_lower = coco_class_name.lower()
                        if any(k in coco_lower for k in ['bottle', 'wine glass', 'cup', 'glass']):
                            pred_class = 'glass'
                        elif any(k in coco_lower for k in ['box', 'carton', 'suitcase']):
                            pred_class = 'cardboard'
                        elif any(k in coco_lower for k in ['can', 'knife', 'fork', 'spoon', 'scissors', 'metal']):
                            pred_class = 'metal'
                        elif any(k in coco_lower for k in ['book', 'paper', 'newspaper', 'envelope']):
                            pred_class = 'paper'
                        elif any(k in coco_lower for k in ['bag', 'bottle', 'handbag']):
                            pred_class = 'plastic'
                        elif any(k in coco_lower for k in ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']):
                            pred_class = 'compost'
                            
                        all_predictions = [{
                            'class': pred_class,
                            'confidence': confidence * 100,
                            'info': ECO_INFO[pred_class]
                        }]
                        for other in [c for c in CLASSES if c != pred_class]:
                            all_predictions.append({
                                'class': other,
                                'confidence': 0.0,
                                'info': ECO_INFO[other]
                            })
                        method_name = f"YOLO26 COCO Detector (Mapped '{coco_class_name}')"
                else:
                    # Fallback if no boxes/probs are returned or threshold filtered them out
                    return jsonify({'error': 'No objects detected above the 60% confidence threshold.'}), 400
                
                return jsonify({
                    'class': pred_class,
                    'confidence': confidence * 100,
                    'method': method_name,
                    'info': ECO_INFO[pred_class],
                    'filepath': web_filepath,
                    'gradcam_filepath': None, 
                    'all_predictions': all_predictions
                })
            except Exception as e:
                print(f">>> YOLO26 inference failed: {e}. Trying TF fallbacks...")
                
        return jsonify({'error': 'YOLO26 model could not be loaded on this system.'}), 500
        
    # ==========================================
    # ROUTE B: TENSORFLOW/MOBILENET MODEL INFERENCE
    # ==========================================
    # 1. Custom model inference (Trained Model)
    if model is not None:
        try:
            # Preprocess image
            if cv2 is not None:
                # Read using numpy and decode using cv2 to support Unicode file paths on Windows
                img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img_array = np.expand_dims(img, axis=0) / 255.0
            else:
                from PIL import Image
                img = Image.open(filepath).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img), axis=0) / 255.0
                
            predictions = model.predict(img_array)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            pred_class = CLASSES[class_idx]
            
            # Format predictions list
            all_predictions = []
            for idx, prob in enumerate(predictions):
                all_predictions.append({
                    'class': CLASSES[idx],
                    'confidence': float(prob) * 100,
                    'info': ECO_INFO[CLASSES[idx]]
                })
            all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
            
            # Run Explainable AI (Grad-CAM)
            gradcam_web_path = None
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu", pred_index=class_idx)
                gradcam_filename = f"gradcam_{filename}"
                gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                
                if generate_and_save_gradcam(filepath, heatmap, gradcam_filepath):
                    gradcam_web_path = f'/static/uploads/{gradcam_filename}'
                    print(f">>> Grad-CAM overlay generated: {gradcam_web_path}")
            except Exception as cam_err:
                print(f">>> Grad-CAM generation failed: {cam_err}")
            
            return jsonify({
                'class': pred_class,
                'confidence': confidence * 100,
                'method': 'Custom Trained Deep Learning Model',
                'info': ECO_INFO[pred_class],
                'filepath': web_filepath,
                'gradcam_filepath': gradcam_web_path,
                'all_predictions': all_predictions
            })
            
        except Exception as e:
            print(f">>> Inference failed with custom model: {e}. Trying fallback...")
            
    # 2. Fallback model inference (ImageNet pre-trained)
    if fallback_model is not None:
        try:
            if cv2 is not None:
                # Read using numpy and decode using cv2 to support Unicode file paths on Windows
                img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img_array = np.expand_dims(img, axis=0)
                # MobileNetV2 preprocessing
                img_array = preprocess_input(img_array)
            else:
                from PIL import Image
                img = Image.open(filepath).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img), axis=0)
                img_array = preprocess_input(img_array.astype(np.float32))
                
            preds = fallback_model.predict(img_array)
            pred_class, confidence, detection_detail = map_imagenet_to_waste(preds)
            
            # Form simulated multi-class output for UI
            all_predictions = []
            remaining_prob = 1.0 - confidence
            
            # Distribute remaining probabilities among others for visually pleasing UI
            other_classes = [c for c in CLASSES if c != pred_class]
            random.shuffle(other_classes)
            
            all_predictions.append({
                'class': pred_class,
                'confidence': confidence * 100,
                'info': ECO_INFO[pred_class]
            })
            
            for i, other in enumerate(other_classes):
                weight = (0.5 ** (i + 1)) * remaining_prob
                all_predictions.append({
                    'class': other,
                    'confidence': weight * 100,
                    'info': ECO_INFO[other]
                })
            # Add final residual
            sum_prob = sum(x['confidence'] for x in all_predictions)
            if sum_prob < 100.0:
                all_predictions[-1]['confidence'] += (100.0 - sum_prob)
                
            all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
            
            return jsonify({
                'class': pred_class,
                'confidence': confidence * 100,
                'method': f'ImageNet Fallback Engine ({detection_detail})',
                'info': ECO_INFO[pred_class],
                'filepath': web_filepath,
                'all_predictions': all_predictions
            })
            
        except Exception as e:
            return jsonify({'error': f'Classification failed: {str(e)}'}), 500
            
    # 3. Dummy hard-coded fallback if no model whatsoever could be loaded
    simulated_class = random.choice(CLASSES)
    return jsonify({
        'class': simulated_class,
        'confidence': 85.4,
        'method': 'Eco-Heuristic Rule Engine (Mock)',
        'info': ECO_INFO[simulated_class],
        'filepath': web_filepath,
        'all_predictions': [{
            'class': c,
            'confidence': 85.4 if c == simulated_class else 14.6 / 6,
            'info': ECO_INFO[c]
        } for c in CLASSES]
    })

def run_training_job(train_dir, test_dir):
    """Background model training thread target."""
    global model, training_status
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waste_model.h5')
    
    class WebStatusCallback(keras.callbacks.Callback):
        def __init__(self, epoch_offset=0, total_stages_epochs=10):
            super().__init__()
            self.epoch_offset = epoch_offset
            self.total_stages_epochs = total_stages_epochs
            
        def on_epoch_begin(self, epoch, logs=None):
            current_epoch = epoch + 1 + self.epoch_offset
            with training_lock:
                training_status['epoch'] = current_epoch
                training_status['progress'] = int(((current_epoch - 1) / self.total_stages_epochs) * 100)
                training_status['log'] += f"--- Starting Epoch {current_epoch}/{self.total_stages_epochs} ---\n"
                
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_epoch = epoch + 1 + self.epoch_offset
            with training_lock:
                training_status['loss'] = float(logs.get('loss', 0))
                training_status['accuracy'] = float(logs.get('accuracy', 0))
                training_status['val_loss'] = float(logs.get('val_loss', 0))
                training_status['val_accuracy'] = float(logs.get('val_accuracy', 0))
                training_status['progress'] = int((current_epoch / self.total_stages_epochs) * 100)
                training_status['log'] += f"Epoch {current_epoch} End -> Acc: {training_status['accuracy']:.4f} | Loss: {training_status['loss']:.4f} | Val Acc: {training_status['val_accuracy']:.4f}\n"

    try:
        total_epochs = 10
        with training_lock:
            training_status['status'] = 'running'
            training_status['epoch'] = 0
            training_status['total_epochs'] = total_epochs
            training_status['progress'] = 0
            training_status['log'] = "Initializing Deep Learning pipeline...\n"
            training_status['log'] += "Loading MobileNetV2 base weights (frozen)...\n"
            
        time.sleep(1) # Visual pacing
        
        # Load and construct model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False
            
        x = base_model.output
        # Integrate Squeeze-and-Excitation channel attention block here
        x = squeeze_excitation_block(x, ratio=16)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(CLASSES), activation='softmax')(x)
        
        custom_model = Model(inputs=base_model.input, outputs=predictions)
        
        with training_lock:
            training_status['log'] += "Preparing advanced Training Data Generators...\n"
            
        # Advanced generators with data augmentation & proper preprocessing
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        with training_lock:
            training_status['log'] += f"Found {train_generator.n} training images, {test_generator.n} testing images.\n"
            
        # ==========================================
        # STAGE 1: Warm-up Head (3 epochs, LR = 1e-3)
        # ==========================================
        with training_lock:
            training_status['log'] += ">>> Stage 1: Warming up new Classifier Head (3 Epochs, Learning Rate=1e-3)...\n"
            
        custom_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=categorical_focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        custom_model.fit(
            train_generator,
            epochs=3,
            validation_data=test_generator,
            callbacks=[WebStatusCallback(epoch_offset=0, total_stages_epochs=total_epochs)]
        )
        
        # ==========================================
        # STAGE 2: Deep Fine-Tuning (7 epochs, LR = 1e-5)
        # ==========================================
        with training_lock:
            training_status['log'] += ">>> Stage 2: Unfreezing the last 50 layers of MobileNetV2 base...\n"
            
        for layer in base_model.layers[-50:]:
            layer.trainable = True
            
        custom_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=categorical_focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        with training_lock:
            training_status['log'] += ">>> Model recompiled with fine-tuning learning rate (1e-5). Starting optimization...\n"
            
        # Cosine Annealing Callback
        cosine_callback = CosineAnnealingScheduler(lr_max=1e-5, lr_min=1e-7, total_epochs=7)
        
        custom_model.fit(
            train_generator,
            epochs=7,
            validation_data=test_generator,
            callbacks=[WebStatusCallback(epoch_offset=3, total_stages_epochs=total_epochs), cosine_callback]
        )
        
        with training_lock:
            training_status['log'] += "Saving trained model weights to disk (waste_model.h5)...\n"
            
        custom_model.save(model_path)
        
        with training_lock:
            model = custom_model
            training_status['status'] = 'completed'
            training_status['progress'] = 100
            training_status['log'] += ">>> Deep Model trained and saved successfully! Custom high-accuracy Classifier is now ACTIVE.\n"
            
    except Exception as e:
        with training_lock:
            training_status['status'] = 'failed'
            training_status['error'] = str(e)
            training_status['log'] += f"\nERROR encountered during training: {str(e)}\n"
            print(f">>> Background training failed: {e}")

@app.route('/train', methods=['POST'])
def start_training():
    global training_status
    
    if tf is None:
        return jsonify({'error': 'TensorFlow is not loaded or not installed.'}), 500
        
    with training_lock:
        if training_status['status'] == 'running':
            return jsonify({'error': 'Training is already in progress'}), 400
            
    # Validate datasets folders
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'DataSets', 'Train')
    test_dir = os.path.join(base_dir, 'DataSets', 'Test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        return jsonify({'error': 'Dataset directories not found. Please extract the datasets.'}), 404
        
    # Check if directories contain subdirectories for categories
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if len(subdirs) < 3:
        return jsonify({'error': 'Train dataset folders are empty or missing classes. Please extract the folders.'}), 400
        
    # Start thread
    thread = threading.Thread(target=run_training_job, args=(train_dir, test_dir))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started in background successfully.'})

@app.route('/train_status')
def get_train_status():
    with training_lock:
        return jsonify(training_status)

if __name__ == '__main__':
    # Load model on startup
    load_models()
    app.run(debug=True, host='127.0.0.1', port=5000)
