from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import warnings
import sys
import logging
import base64
import io
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to save memory
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
def load_keras_model():
    try:
        model_path = 'models/inception_resnet_best_model.keras'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None
        # Load the original Inception-ResNet-v2 model
        model = load_model(model_path)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Initialize model at startup
model = load_keras_model()

# Class information with detailed medical descriptions
class_info = {
    0: {
        "name": "Normal (KL Grade 0)", 
        "desc": "No signs of osteoarthritis. Healthy knee joint with normal joint space and no osteophytes.",
        "color": "#00ff88", 
        "icon": "🟢",
        "symptoms": "No symptoms, normal knee function",
        "treatment": "Maintain healthy lifestyle, regular exercise",
        "risk": "Low risk of progression"
    },
    1: {
        "name": "Doubtful (KL Grade 1)", 
        "desc": "Possible minimal osteophytes, uncertain joint space narrowing. Very early signs of OA.",
        "color": "#ffff00", 
        "icon": "🟡",
        "symptoms": "Mild discomfort, occasional stiffness",
        "treatment": "Physical therapy, weight management, anti-inflammatory diet",
        "risk": "Moderate risk, monitor progression"
    },
    2: {
        "name": "Mild (KL Grade 2)", 
        "desc": "Definite osteophytes, possible joint space narrowing. Early osteoarthritis confirmed.",
        "color": "#ff8800", 
        "icon": "🟠",
        "symptoms": "Pain with activity, morning stiffness, reduced range of motion",
        "treatment": "Physical therapy, pain management, joint protection exercises",
        "risk": "High risk, active intervention needed"
    },
    3: {
        "name": "Moderate (KL Grade 3)", 
        "desc": "Multiple osteophytes, definite joint space narrowing. Moderate osteoarthritis.",
        "color": "#ff4400", 
        "icon": "🔴",
        "symptoms": "Significant pain, limited mobility, joint instability",
        "treatment": "Advanced pain management, assistive devices, surgical consultation",
        "risk": "Very high risk, aggressive treatment required"
    },
    4: {
        "name": "Severe (KL Grade 4)", 
        "desc": "Large osteophytes, severe joint space narrowing, bone deformation. Advanced osteoarthritis.",
        "color": "#ff0000", 
        "icon": "💀",
        "symptoms": "Severe pain, significant disability, joint deformity",
        "treatment": "Surgical intervention (knee replacement), comprehensive pain management",
        "risk": "Critical, immediate surgical evaluation needed"
    }
}

# Image preprocessing
def preprocess_img(img, target_size=(224, 224)):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Grad-CAM functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def create_gradcam_overlay(img, heatmap, alpha=0.7):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Create red heatmap for knee OA visualization
    heatmap_colored = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    heatmap_colored[:, :, 0] = heatmap  # Red channel
    heatmap_colored[:, :, 1] = heatmap // 3  # Green channel (darker)
    heatmap_colored[:, :, 2] = heatmap // 3  # Blue channel (darker)
    
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

def generate_gradcam(model, img_array, original_img, last_conv_layer_name='conv_7b_ac'):
    try:
        # Get the base model from the custom model
        base_model = model.get_layer('inception_resnet_v2')
        
        # Make predictions
        preds = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(preds[0])
        
        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, predicted_class)
        
        # Create overlay
        overlay_img = create_gradcam_overlay(original_img, heatmap)
        
        return overlay_img, predicted_class, preds[0][predicted_class]
    except Exception as e:
        logger.error(f"Grad-CAM error: {str(e)}")
        return original_img, 0, 0.0

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/grades')
def grades():
    return render_template('grades.html')

@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        img = Image.open(file.stream)
        img_array = np.array(img)
        original_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        processed_img = preprocess_img(img_array)
        
        # Make prediction
        start_time = time.time()
        predictions = model.predict(processed_img, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = predictions[0][pred_class]
        
        # Generate Grad-CAM
        gradcam_img, gradcam_class, gradcam_confidence = generate_gradcam(
            model, processed_img, original_img
        )
        
        # Convert images to base64 for frontend
        def image_to_base64(img_array):
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        original_b64 = image_to_base64(img_array)
        gradcam_b64 = image_to_base64(gradcam_img)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Prepare response with detailed class information
        result = {
            'success': True,
            'predicted_class': pred_class,
            'class_name': class_info[pred_class]['name'],
            'description': class_info[pred_class]['desc'],
            'symptoms': class_info[pred_class]['symptoms'],
            'treatment': class_info[pred_class]['treatment'],
            'risk': class_info[pred_class]['risk'],
            'confidence': float(confidence),
            'color': class_info[pred_class]['color'],
            'icon': class_info[pred_class]['icon'],
            'processing_time': round(processing_time, 2),
            'confidence_distribution': predictions[0].tolist(),
            'original_image': original_b64,
            'gradcam_image': gradcam_b64
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 5000))
    # Run in production mode when deployed
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)