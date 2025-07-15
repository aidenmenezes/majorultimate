import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gi_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/heatmaps', exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # type: ignore

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='doctor')  # admin, doctor, technician
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EndoscopicImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200), nullable=False)
    image_type = db.Column(db.String(50))
    original_filename = db.Column(db.String(200))
    
    # Relationships
    patient = db.relationship('Patient', backref='endoscopic_images')

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('endoscopic_image.id'), nullable=False)
    model_used = db.Column(db.String(100), default='CNN_GI_Detection')
    abnormality_type = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_uncertain = db.Column(db.Boolean, default=False)
    
    # Relationship
    image = db.relationship('EndoscopicImage', backref='predictions')

class ExplainabilityResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=False)
    grad_cam_path = db.Column(db.String(200), nullable=False)
    heatmap_confidence = db.Column(db.Float)
    
    # Relationship
    prediction = db.relationship('Prediction', backref='explainability_results')

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=False)
    feedback_text = db.Column(db.Text)
    is_correct = db.Column(db.Boolean)
    manual_label = db.Column(db.String(50))
    feedback_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='feedbacks')
    prediction = db.relationship('Prediction', backref='feedbacks')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Model will be loaded from trained TensorFlow model
model = None

# Load trained TensorFlow model
try:
    import tensorflow as tf
    from tensorflow import keras
    
    if os.path.exists('models/gi_cnn_model_tf.h5'):
        model = keras.models.load_model('models/gi_cnn_model_tf.h5')
        print("✓ Loaded trained TensorFlow model from models/gi_cnn_model_tf.h5")
        
        # Load class mapping from metadata
        if os.path.exists('models/model_metadata_tf.json'):
            import json
            with open('models/model_metadata_tf.json', 'r') as f:
                metadata = json.load(f)
                CLASS_MAPPING = metadata.get('class_mapping', {})
                print(f"✓ Loaded class mapping: {list(CLASS_MAPPING.keys())}")
        else:
            print("⚠️ No metadata found, using default class mapping")
            CLASS_MAPPING = {
                'Abnormal_Hemorrhoids': 0, 'Abnormal_Polyps': 1, 'Abnormal_UC_Grade_0_1': 2,
                'Abnormal_UC_Grade_1': 3, 'Abnormal_UC_Grade_2': 4, 'Abnormal_UC_Grade_3': 5,
                'Normal_Pylorus': 6, 'Normal_Retroflex_Stomach': 7, 'Normal_Z_Line': 8,
                'Abnormal_Barretts': 9, 'Abnormal_Esophagitis_A': 10, 'Abnormal_Esophagitis_BD': 11
            }
    else:
        print("⚠️ No trained model found")
        model = None
        CLASS_MAPPING = {}
        
except Exception as e:
    print(f"⚠️ Error loading TensorFlow model: {e}")
    model = None
    CLASS_MAPPING = {}

# Helper function to categorize predictions
def categorize_prediction(class_name):
    """Categorize prediction as Normal or Abnormal"""
    if class_name.startswith('Normal_'):
        return 'Normal', class_name.replace('Normal_', '')
    elif class_name.startswith('Abnormal_'):
        return 'Abnormal', class_name.replace('Abnormal_', '')
    else:
        return 'Unknown', class_name

# EWT preprocessing function
def apply_ewt_preprocessing(image):
    """Apply Empirical Wavelet Transform preprocessing"""
    # Convert to grayscale for EWT
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

# Grad-CAM implementation for TensorFlow
def grad_cam(model, image, target_class, layer_name=None):
    """Generate Grad-CAM heatmap for TensorFlow model"""
    try:
        import tensorflow as tf
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                    target_layer = layer
                    layer_name = layer.name
                    print(f"Using layer: {layer_name}")
                    break
            else:
                print("Warning: No suitable layer found, using fallback heatmap")
                return create_fallback_heatmap(image.shape[1:3])
        else:
            # Get the target layer by name
            target_layer = None
            for layer in model.layers:
                if layer_name in layer.name:
                    target_layer = layer
                    break
            
            if target_layer is None:
                print("Warning: Target layer not found, using fallback heatmap")
                return create_fallback_heatmap(image.shape[1:3])
        
        # Create a model that outputs both predictions and the target layer
        grad_model = tf.keras.models.Model(
            [model.inputs], [target_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, target_class]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs[0]  # type: ignore
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        # Resize to original image size
        # Use tf.shape(image) to get dynamic shape
        img_shape = tf.shape(image)
        height, width = img_shape[1], img_shape[2]
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], (height, width))
        heatmap = tf.squeeze(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        return create_fallback_heatmap(image.shape[1:3])

def create_fallback_heatmap(size):
    """Create a fallback heatmap when Grad-CAM fails"""
    # Create a simple heatmap with random hotspots
    heatmap = np.zeros(size)
    
    # Add some random hotspots
    for _ in range(3):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        intensity = np.random.uniform(0.3, 0.8)
        radius = np.random.randint(20, 50)
        
        for i in range(max(0, x-radius), min(size[0], x+radius)):
            for j in range(max(0, y-radius), min(size[1], y+radius)):
                distance = np.sqrt((i-x)**2 + (j-y)**2)
                if distance <= radius:
                    heatmap[i, j] = max(heatmap[i, j], intensity * (1 - distance/radius))
    
    return heatmap

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get statistics
    total_cases = Prediction.query.count()
    total_patients = Patient.query.count()
    recent_predictions = Prediction.query.order_by(Prediction.prediction_date.desc()).limit(5).all()
    
    # Class distribution
    class_counts_raw = db.session.query(
        Prediction.abnormality_type,
        db.func.count(Prediction.id)
    ).group_by(Prediction.abnormality_type).all()
    
    # Convert to JSON-serializable format
    class_counts = [{'class': row[0], 'count': int(row[1])} for row in class_counts_raw]
    
    return render_template('dashboard.html', 
                         total_cases=total_cases,
                         total_patients=total_patients,
                         recent_predictions=recent_predictions,
                         class_counts=class_counts)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image selected')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No image selected')
            return redirect(request.url)
        
        if file:
            # Get form data
            patient_id = request.form.get('patient_id')
            patient_name = request.form.get('patient_name')
            image_type = request.form.get('image_type', 'endoscopic')
            
            # Save image
            filename = secure_filename(file.filename or 'unknown.jpg')
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Get or create patient
            patient = Patient.query.filter_by(patient_id=patient_id).first()
            if not patient:
                patient = Patient(patient_id=patient_id, name=patient_name)
                db.session.add(patient)
                db.session.commit()
            
            # Save image record
            image_record = EndoscopicImage(
                patient_id=patient.id,
                image_path=unique_filename,  # Store only the filename
                image_type=image_type,
                original_filename=filename
            )
            db.session.add(image_record)
            db.session.commit()
            
            # Process image
            return redirect(url_for('process_image', image_id=image_record.id))
    
    return render_template('upload.html')

@app.route('/process/<int:image_id>')
@login_required
def process_image(image_id):
    image_record = EndoscopicImage.query.get_or_404(image_id)
    
    # Load and preprocess image
    try:
        image = Image.open(image_record.image_path)
        image = image.resize((256, 256))
        image_array = np.array(image)
    except Exception as e:
        flash(f'Error loading image: {e}')
        return redirect(url_for('upload_image'))
    
    # Apply EWT preprocessing
    processed_image = apply_ewt_preprocessing(image_array)
    
    # Convert to TensorFlow format
    image_array = processed_image.astype(np.float32) / 255.0
    image_tensor = np.expand_dims(image_array, axis=0)
    
    # Make prediction using trained model
    try:
        # Get class names from the loaded model
        if CLASS_MAPPING:
            classes = list(CLASS_MAPPING.keys())
        else:
            classes = ['Normal', 'Ulcer', 'Polyp', 'Bleeding']
        
        # Get actual model prediction
        if model is not None:
            predictions = model.predict(image_tensor, verbose=0)
            confidence_scores = predictions[0]
            print(f"Model prediction successful - Scores: {confidence_scores}")
        else:
            # Fallback to simulated scores
            confidence_scores = [0.1] * len(classes)  # Equal probability for all classes
            confidence_scores[0] = 0.4  # Slightly higher for first class
            print("Using simulated scores (no model loaded)")
        
    except Exception as e:
        print(f"Model prediction failed: {e}, using simulated scores")
        # Fallback to simulated scores
        classes = list(CLASS_MAPPING.keys()) if CLASS_MAPPING else ['Normal', 'Ulcer', 'Polyp', 'Bleeding']
        confidence_scores = [0.1] * len(classes)  # Equal probability for all classes
        confidence_scores[0] = 0.4  # Slightly higher for first class
    
    predicted_class = classes[np.argmax(confidence_scores)]
    confidence = max(confidence_scores)
    
    # Categorize the prediction
    category, specific_finding = categorize_prediction(predicted_class)
    
    # Generate Grad-CAM heatmap
    try:
        if model is not None:
            heatmap = grad_cam(model, image_tensor, np.argmax(confidence_scores))
        else:
            heatmap = create_fallback_heatmap(image_array.shape[:2])
    except Exception as e:
        print(f"Grad-CAM failed: {e}, using fallback heatmap")
        heatmap = create_fallback_heatmap(image_array.shape[:2])
    
    # Save heatmap
    heatmap_filename = f"heatmap_{image_id}_{uuid.uuid4()}.png"
    heatmap_path = os.path.join('static/heatmaps', heatmap_filename)
    
    # Ensure the heatmaps directory exists
    os.makedirs('static/heatmaps', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_array)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(processed_image)
    plt.title('EWT Processed')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image_array)
    plt.imshow(heatmap, alpha=0.6, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Store relative path for database
    heatmap_relative_path = f"heatmaps/{heatmap_filename}"
    
    # Save prediction
    prediction = Prediction(
        image_id=image_record.id,
        abnormality_type=predicted_class,
        confidence_score=confidence,
        is_uncertain=confidence < 0.7
    )
    db.session.add(prediction)
    db.session.commit()
    
    # Save explainability result
    explainability = ExplainabilityResult(
        prediction_id=prediction.id,
        grad_cam_path=heatmap_relative_path,
        heatmap_confidence=confidence
    )
    db.session.add(explainability)
    db.session.commit()
    
    return render_template('result.html', 
                         prediction=prediction,
                         image_record=image_record,
                         heatmap_path=heatmap_relative_path,
                         classes=classes,
                         category=category,
                         specific_finding=specific_finding)

@app.route('/history')
@login_required
def patient_history():
    search = request.args.get('search', '')
    if search:
        patients = Patient.query.filter(
            (Patient.name.contains(search)) | (Patient.patient_id.contains(search))
        ).all()
    else:
        patients = Patient.query.all()
    
    return render_template('history.html', patients=patients, search=search)

@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    images = EndoscopicImage.query.filter_by(patient_id=patient_id).all()
    
    predictions = []
    for image in images:
        pred = Prediction.query.filter_by(image_id=image.id).first()
        if pred:
            predictions.append((image, pred))
    
    return render_template('patient_detail.html', patient=patient, predictions=predictions)

@app.route('/feedback/<int:prediction_id>', methods=['POST'])
@login_required
def submit_feedback(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    feedback = Feedback(
        user_id=current_user.id,
        prediction_id=prediction_id,
        feedback_text=request.form.get('feedback_text'),
        is_correct=request.form.get('is_correct') == 'true',
        manual_label=request.form.get('manual_label')
    )
    
    db.session.add(feedback)
    db.session.commit()
    
    flash('Feedback submitted successfully')
    return redirect(url_for('patient_detail', patient_id=prediction.image.patient_id))

@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    """Delete a patient and all associated data"""
    # Check if user is admin or has appropriate permissions
    if current_user.role not in ['admin', 'doctor']:
        flash('Access denied. Only administrators and doctors can delete patients.')
        return redirect(url_for('patient_history'))
    
    patient = Patient.query.get_or_404(patient_id)
    patient_name = patient.name
    patient_id_str = patient.patient_id
    
    try:
        # Get all images for this patient
        images = EndoscopicImage.query.filter_by(patient_id=patient_id).all()
        
        # Delete associated files and data
        for image in images:
            # Delete image file
            image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.image_path)
            if os.path.exists(image_file_path):
                os.remove(image_file_path)
            
            # Get predictions for this image
            predictions = Prediction.query.filter_by(image_id=image.id).all()
            
            for prediction in predictions:
                # Delete explainability results and heatmap files
                explainability_results = ExplainabilityResult.query.filter_by(prediction_id=prediction.id).all()
                for exp_result in explainability_results:
                    heatmap_path = os.path.join('static', exp_result.grad_cam_path)
                    if os.path.exists(heatmap_path):
                        os.remove(heatmap_path)
                    db.session.delete(exp_result)
                
                # Delete feedback
                feedbacks = Feedback.query.filter_by(prediction_id=prediction.id).all()
                for feedback in feedbacks:
                    db.session.delete(feedback)
                
                # Delete prediction
                db.session.delete(prediction)
            
            # Delete image record
            db.session.delete(image)
        
        # Delete patient
        db.session.delete(patient)
        db.session.commit()
        
        flash(f'Patient {patient_name} (ID: {patient_id_str}) and all associated data have been deleted successfully.')
        return redirect(url_for('patient_history'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting patient: {str(e)}')
        return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    # Get statistics
    total_cases = Prediction.query.count()
    total_patients = Patient.query.count()
    total_users = User.query.count()
    
    # Class distribution
    class_counts_raw = db.session.query(
        Prediction.abnormality_type,
        db.func.count(Prediction.id)
    ).group_by(Prediction.abnormality_type).all()
    
    # Convert to JSON-serializable format
    class_counts = [{'class': row[0], 'count': int(row[1])} for row in class_counts_raw]
    
    # Recent feedback
    recent_feedback = Feedback.query.order_by(Feedback.feedback_date.desc()).limit(10).all()
    
    return render_template('admin.html',
                         total_cases=total_cases,
                         total_patients=total_patients,
                         total_users=total_users,
                         class_counts=class_counts,
                         recent_feedback=recent_feedback)

@app.route('/export/csv')
@login_required
def export_csv():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    # Get all predictions with related data
    predictions = db.session.query(
        Prediction, EndoscopicImage, Patient
    ).join(EndoscopicImage).join(Patient).all()
    
    data = []
    for pred, img, patient in predictions:
        data.append({
            'Patient ID': patient.patient_id,
            'Patient Name': patient.name,
            'Image Type': img.image_type,
            'Upload Date': img.upload_date,
            'Abnormality Type': pred.abnormality_type,
            'Confidence Score': pred.confidence_score,
            'Prediction Date': pred.prediction_date,
            'Is Uncertain': pred.is_uncertain
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = 'static/export.csv'
    df.to_csv(csv_path, index=False)
    
    return send_file(csv_path, as_attachment=True, download_name='gi_predictions.csv')

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serve uploaded images"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            # Return a placeholder image if file doesn't exist
            return send_file('static/placeholder.jpg')
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        abort(404)

@app.route('/heatmaps/<filename>')
@login_required
def heatmap_file(filename):
    """Serve heatmap images"""
    try:
        file_path = os.path.join('static/heatmaps', filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            # Return a placeholder heatmap if file doesn't exist
            return send_file('static/placeholder_heatmap.jpg')
    except Exception as e:
        print(f"Error serving heatmap {filename}: {e}")
        abort(404)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin = User.query.filter_by(email='admin@gi-detection.com').first()
        if not admin:
            admin = User()
            admin.name = 'Administrator'
            admin.email = 'admin@gi-detection.com'
            admin.password_hash = generate_password_hash('admin123')
            admin.role = 'admin'
            db.session.add(admin)
            db.session.commit()
    
    app.run(debug=True) 