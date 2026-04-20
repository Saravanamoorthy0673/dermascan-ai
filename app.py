import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import io
import json
import numpy as np
import cv2
import tensorflow as tf
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///dermascan.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
from model_download import download_model_if_needed
download_model_if_needed()

# ─── CONFIG ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER']                  = 'static/uploads'
app.config['HEATMAP_FOLDER']                 = 'static/heatmaps'
app.config['MAX_CONTENT_LENGTH']             = 10 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dermascan-secret-key-2024')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

db = SQLAlchemy(app)
    
MODEL_PATH = os.path.join(os.getcwd(), "model", "skin_disease_model.h5")
IMAGE_SIZE  = (224, 224)
CLASS_NAMES = ['Acne', 'Melanoma', 'Psoriasis', 'Rosacea', 'Vitiligo']

# ─── MC DROPOUT CONFIG ────────────────────────────────────────────
# Number of stochastic forward passes
# அதிகமா போட்டா accurate, ஆனா slow — 20 ideal
MC_RUNS = 1

# Uncertainty thresholds (percentage)
# இந்த values tune பண்ணலாம் உன் model-க்கு ஏத்தாப்போல
UNCERTAINTY_LOW    = 5.0   # Below this  → High Confidence   (green)
UNCERTAINTY_MEDIUM = 15.0  # Below this  → Moderate          (orange)
                           # Above 15    → Low Confidence     (red + warning)

DISEASE_INFO = {
    'Acne': {
        'description': 'A common skin condition causing pimples, blackheads, and inflammation, usually on the face.',
        'causes':      'Excess oil production, clogged hair follicles, bacteria, hormonal changes.',
        'symptoms':    'Pimples, blackheads, whiteheads, oily skin, scarring.',
        'treatment':   'Use mild cleanser, avoid oily cosmetics, apply benzoyl peroxide. Consult dermatologist if severe.',
        'severity':    'Mild to Moderate',
        'keywords':    ['pimple', 'acne', 'blackhead', 'whitehead', 'zit', 'breakout',
                        'oily face', 'face bump', 'blemish', 'clogged pore', 'pus', 'acne scar'],
    },
    'Melanoma': {
        'description': 'A serious form of skin cancer that develops in melanocytes (pigment-producing cells).',
        'causes':      'UV radiation exposure, genetic factors, history of sunburn.',
        'symptoms':    'Unusual mole, change in existing mole, dark lesion on skin.',
        'treatment':   'Seek a dermatologist IMMEDIATELY. Early detection is critical for survival.',
        'severity':    'Severe — Urgent Medical Attention Required',
        'keywords':    ['melanoma', 'irregular mole', 'changing mole', 'dark lesion',
                        'asymmetric mole', 'black spot', 'dark growth', 'skin cancer',
                        'mole changed', 'bleeding mole', 'uneven mole'],
    },
    'Psoriasis': {
        'description': 'A chronic autoimmune condition causing rapid skin cell buildup leading to scaling.',
        'causes':      'Immune system dysfunction, genetic factors, stress, skin injury.',
        'symptoms':    'Red patches, silver scales, dry cracked skin, itching, burning.',
        'treatment':   'Medicated moisturizers, topical corticosteroids, avoid skin irritants. See a dermatologist.',
        'severity':    'Moderate to Chronic',
        'keywords':    ['psoriasis', 'scaly skin', 'dry scaly', 'silver scale', 'skin plaque',
                        'flaky skin', 'itchy patch', 'red scaly', 'thick patch', 'skin flakes',
                        'dry cracked', 'scaling skin', 'elbow patch', 'knee patch', 'scalp flakes'],
    },
    'Rosacea': {
        'description': 'A chronic skin condition causing facial redness and visible blood vessels.',
        'causes':      'Genetics, environmental triggers, sun exposure, spicy food, alcohol.',
        'symptoms':    'Facial redness, swollen red bumps, eye problems, enlarged nose.',
        'treatment':   'Avoid triggers, use sunscreen daily, gentle skincare. Doctor may prescribe antibiotics.',
        'severity':    'Mild to Moderate',
        'keywords':    ['rosacea', 'facial redness', 'red face', 'face flush', 'cheek redness',
                        'nose redness', 'face bumps', 'skin flushing', 'burning face',
                        'red cheeks', 'visible veins', 'face blush'],
    },
    'Vitiligo': {
        'description': 'A condition where skin loses pigment in patches due to destroyed melanocytes.',
        'causes':      'Autoimmune disorder, genetic factors, stress, skin trauma.',
        'symptoms':    'White patches on skin, premature whitening of hair, loss of color in tissues.',
        'treatment':   'Topical corticosteroids, light therapy, skin camouflage. Consult a dermatologist.',
        'severity':    'Mild to Moderate',
        'keywords':    ['vitiligo', 'white patch', 'white spot', 'skin depigment', 'color loss',
                        'pale patch', 'loss of skin color', 'white skin patch', 'light skin patch',
                        'skin whitening', 'pigment loss', 'pale spot'],
    },
}

# ─── DATABASE MODELS ──────────────────────────────────────────────
class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.String(30),  nullable=False)
    scans      = db.relationship('ScanRecord', backref='user', lazy=True)

class ScanRecord(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    episode_id       = db.Column(db.Integer, db.ForeignKey('episode.id'), nullable=True)
    date             = db.Column(db.String(20),  nullable=False)
    time             = db.Column(db.String(20),  nullable=False)
    disease          = db.Column(db.String(50),  nullable=False)
    confidence       = db.Column(db.Float,       nullable=False)
    uncertainty      = db.Column(db.Float,       nullable=True)   # ← NEW: MC Dropout uncertainty
    uncertainty_level= db.Column(db.String(30),  nullable=True)   # ← NEW: High/Moderate/Low
    severity         = db.Column(db.String(100), nullable=True)
    symptoms         = db.Column(db.String(500), nullable=True)
    image_file       = db.Column(db.String(200), nullable=True)
    heatmap_file     = db.Column(db.String(200), nullable=True)

    def to_dict(self):
        return {
            'id':               self.id,
            'date':             self.date,
            'time':             self.time,
            'disease':          self.disease,
            'confidence':       self.confidence,
            'uncertainty':      self.uncertainty,       # ← NEW
            'uncertainty_level':self.uncertainty_level, # ← NEW
            'severity':         self.severity,
            'symptoms':         self.symptoms,
            'episode_id':       self.episode_id,
            'image_url':        f'/static/uploads/{self.image_file}'    if self.image_file   else None,
            'heatmap_url':      f'/static/heatmaps/{self.heatmap_file}' if self.heatmap_file else None,
        }

class Episode(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease    = db.Column(db.String(50),  nullable=False)
    start_date = db.Column(db.String(30),  nullable=False)
    end_date   = db.Column(db.String(30),  nullable=True)
    status     = db.Column(db.String(20),  nullable=False, default='active')
    notes      = db.Column(db.String(300), nullable=True)
    scans      = db.relationship('ScanRecord', backref='episode', lazy=True)

    def to_dict(self):
        return {
            'id':         self.id,
            'disease':    self.disease,
            'start_date': self.start_date,
            'end_date':   self.end_date,
            'status':     self.status,
            'notes':      self.notes,
            'scan_count': len(self.scans),
        }


# ✅ AFTER ALL MODELS
with app.app_context():
    db.create_all()
    print("✅ Tables created successfully")


# ─── LOGIN REQUIRED DECORATOR ─────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── LOAD MODEL ───────────────────────────────────────────────────
model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model(MODEL_PATH, compile=False)
    return model
    
def load_skin_model():
    global model
    try:
        model = load_model(MODEL_PATH, compile=False)
        print(f"[OK] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")


# ─── HELPERS ──────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32')
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ═══════════════════════════════════════════════════════════════════
# MC DROPOUT PREDICTION — CORE NEW FUNCTION
# ═══════════════════════════════════════════════════════════════════
def mc_dropout_predict(processed_image, n_runs=MC_RUNS):
    """
    Monte Carlo Dropout Inference.

    Normal prediction: model(x, training=False) → dropout OFF → single result
    MC Dropout:        model(x, training=True)  → dropout ON  → n_runs results

    Same image → slightly different neurons OFF each run → different predictions.
    Mean of all runs = final confidence.
    Std  of all runs = uncertainty (how much the model disagrees with itself).

    Low std  = model is sure every time → High Confidence
    High std = model keeps changing answer → Low Confidence → warn user
    """
    predictions = []

    for _ in range(n_runs):
        # training=True keeps Dropout layers active during inference
        # This is the ONLY change from normal prediction
        model_instance = get_model()
        pred = model_instance(processed_image, training=True)
        predictions.append(pred.numpy())

    # Stack all runs: shape (n_runs, 1, 5)
    predictions = np.array(predictions)

    # Mean across all runs: shape (1, 5)
    mean_pred = np.mean(predictions, axis=0)

    # Std across all runs: shape (1, 5) — this is the uncertainty
    std_pred  = np.std(predictions, axis=0)

    # Get the winning class
    predicted_index = int(np.argmax(mean_pred[0]))
    confidence      = float(mean_pred[0][predicted_index]) * 100
    uncertainty     = float(std_pred[0][predicted_index])  * 100

    # Build full scores for all 5 classes (with uncertainty per class)
    all_scores = []
    for i in range(len(CLASS_NAMES)):
        all_scores.append({
            'disease':     CLASS_NAMES[i],
            'confidence':  round(float(mean_pred[0][i]) * 100, 2),
            'uncertainty': round(float(std_pred[0][i])  * 100, 2),
        })
    all_scores.sort(key=lambda x: x['confidence'], reverse=True)

    return predicted_index, round(confidence, 2), round(uncertainty, 2), all_scores


def get_uncertainty_label(uncertainty):
    """
    Convert raw uncertainty % into a human-readable label + color + warning flag.

    uncertainty < 5%  → High Confidence   (green)  → No warning
    uncertainty < 15% → Moderate          (orange) → No warning
    uncertainty >= 15%→ Low Confidence    (red)    → Show warning to user
    """
    if uncertainty < UNCERTAINTY_LOW:
        return {
            'level':        'High Confidence',
            'color':        'green',
            'show_warning': False,
            'message':      'The AI is confident about this prediction.',
            'badge':        '✅ High Confidence',
        }
    elif uncertainty < UNCERTAINTY_MEDIUM:
        return {
            'level':        'Moderate Confidence',
            'color':        'orange',
            'show_warning': False,
            'message':      'Prediction is reasonably confident. Consider consulting a doctor.',
            'badge':        '⚡ Moderate Confidence',
        }
    else:
        return {
            'level':        'Low Confidence',
            'color':        'red',
            'show_warning': True,
            'message':      'The AI is uncertain about this prediction. Image may be unclear, '
                            'lighting poor, or condition may be complex. Please consult a dermatologist.',
            'badge':        '⚠️ Low Confidence — Please Consult a Doctor',
        }
# ═══════════════════════════════════════════════════════════════════


def predict_disease(img_path):
    """
    Main prediction function — now uses MC Dropout instead of model.predict()
    Returns all original fields + uncertainty fields (backward compatible)
    """
    global model
    import gc

    processed = preprocess_image(img_path)

    # ── MC DROPOUT PREDICTION ──
    predicted_index, confidence, uncertainty, all_scores = mc_dropout_predict(processed)

    predicted_disease = CLASS_NAMES[predicted_index]
    info              = DISEASE_INFO.get(predicted_disease, {})
    uncertainty_info  = get_uncertainty_label(uncertainty)

    result = {
        'predicted_disease': predicted_disease,
        'confidence':        confidence,
        'all_scores':        all_scores,
        'description':       info.get('description', ''),
        'causes':            info.get('causes', ''),
        'symptoms':          info.get('symptoms', ''),
        'treatment':         info.get('treatment', ''),
        'severity':          info.get('severity', ''),
        'uncertainty':         uncertainty,
        'uncertainty_level':   uncertainty_info['level'],
        'uncertainty_color':   uncertainty_info['color'],
        'uncertainty_message': uncertainty_info['message'],
        'uncertainty_badge':   uncertainty_info['badge'],
        'show_warning':        uncertainty_info['show_warning'],
    }

    # ── FREE MEMORY AFTER PREDICTION ──
    model = None
    gc.collect()

    return result
    
def match_symptoms_from_text(text):
    text_lower = text.lower().strip()
    if not text_lower:
        return []
    matches = []
    for disease, info in DISEASE_INFO.items():
        hit_count = 0
        for kw in info.get('keywords', []):
            if kw.lower() in text_lower:
                hit_count += len(kw.split())
        if hit_count > 0:
            matches.append({
                'disease':     disease,
                'match_count': hit_count,
                'description': info['description'],
                'symptoms':    info['symptoms'],
                'treatment':   info['treatment'],
                'severity':    info['severity'],
            })
    matches.sort(key=lambda x: x['match_count'], reverse=True)
    return matches[:2]


def generate_gradcam(img_path, save_path):
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if not last_conv:
        return None
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    img_array = preprocess_image(img_path)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_score = predictions[:, tf.argmax(predictions[0])]
    grads    = tape.gradient(class_score, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap  = tf.squeeze(conv_outputs[0] @ pooled[..., tf.newaxis])
    heatmap  = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap  = heatmap.numpy()
    orig     = cv2.imread(img_path)
    h, w     = orig.shape[:2]
    colored  = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (w, h))), cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(orig, 0.6, colored, 0.4, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)
    return save_path


def generate_pdf_report(result, img_path, heatmap_path, filename, symptoms_text=''):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elems  = []
    T  = ParagraphStyle('T',  parent=styles['Title'],  fontSize=22, textColor=colors.HexColor('#1a3a2a'), alignment=TA_CENTER)
    S  = ParagraphStyle('S',  parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#4a7c59'), alignment=TA_CENTER)
    SE = ParagraphStyle('SE', parent=styles['Normal'], fontSize=12, fontName='Helvetica-Bold', textColor=colors.HexColor('#1a3a2a'), spaceBefore=10, spaceAfter=4)
    B  = ParagraphStyle('B',  parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#333333'), spaceAfter=4, leading=16)
    W  = ParagraphStyle('W',  parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', textColor=colors.HexColor('#c0392b'))
    D  = ParagraphStyle('D',  parent=styles['Normal'], fontSize=8,  textColor=colors.HexColor('#888888'), alignment=TA_CENTER, leading=12)

    elems += [Paragraph("DermaScan AI", T),
              Paragraph("AI-Powered Dermatology Assistant — Medical Report", S),
              Spacer(1, 0.2*cm),
              HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2d6a4f'), spaceAfter=10)]

    now  = datetime.now()
    info = [['Report Date', now.strftime('%d %B %Y')],
            ['Report Time', now.strftime('%I:%M %p')],
            ['Report ID',   f"DS-{now.strftime('%Y%m%d%H%M%S')}"],
            ['Image File',  filename]]
    if symptoms_text:
        info.append(['Symptoms', symptoms_text])

    it = Table(info, colWidths=[4*cm, 12*cm])
    it.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),9),('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
                             ('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#4a7c59')),
                             ('TEXTCOLOR',(1,0),(1,-1),colors.HexColor('#333333')),
                             ('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3)]))
    elems += [it, Spacer(1, 0.3*cm),
              HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=8),
              Paragraph("Images and Grad-CAM Heatmap", SE)]

    ir = []; lr = []
    if os.path.exists(img_path):
        ir.append(RLImage(img_path, width=7*cm, height=7*cm))
        lr.append(Paragraph("Original", ParagraphStyle('IL', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))))
    if heatmap_path and os.path.exists(heatmap_path):
        ir.append(RLImage(heatmap_path, width=7*cm, height=7*cm))
        lr.append(Paragraph("Grad-CAM", ParagraphStyle('IL', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))))
    if ir:
        t = Table([ir, lr], colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER')]))
        elems.append(t)

    conf  = result['confidence']
    cc    = '#2d6a4f' if conf >= 70 else '#e67e22' if conf >= 45 else '#c0392b'

    # ── Uncertainty color for PDF ──
    uc = result.get('uncertainty_color', 'green')
    uc_hex = '#2d6a4f' if uc == 'green' else '#e67e22' if uc == 'orange' else '#c0392b'

    elems += [Spacer(1,0.3*cm), HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=6),
              Paragraph("Diagnosis Result", SE)]

    rt = Table([
        ['Predicted Disease',   result['predicted_disease']],
        ['Confidence Score',    f"{conf:.1f}%"],
        ['Uncertainty Score',   f"{result.get('uncertainty', 0):.1f}%  ({result.get('uncertainty_level','N/A')})"],  # ← NEW row
        ['Severity Level',      result.get('severity','N/A')],
    ], colWidths=[5*cm, 11*cm])

    rt.setStyle(TableStyle([
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#1a3a2a')),
        ('FONTNAME',(1,0),(1,0),'Helvetica-Bold'),
        ('TEXTCOLOR',(1,1),(1,1),colors.HexColor(cc)),
        ('TEXTCOLOR',(1,2),(1,2),colors.HexColor(uc_hex)),   # ← uncertainty row color
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#f4f9f6'),colors.HexColor('#e8f4ee')]),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#cccccc')),
    ]))
    elems.append(rt)

    # ── Show uncertainty warning in PDF if needed ──
    if result.get('show_warning', False):
        W2 = ParagraphStyle('W2', parent=styles['Normal'], fontSize=9,
                            fontName='Helvetica-Bold', textColor=colors.HexColor('#e67e22'))
        elems += [Spacer(1,0.2*cm),
                  Paragraph(f"⚠ UNCERTAINTY WARNING: {result.get('uncertainty_message','')}", W2)]

    if result['predicted_disease'] == 'Melanoma':
        elems += [Spacer(1,0.2*cm), Paragraph("WARNING: Melanoma detected. Consult a dermatologist IMMEDIATELY.", W)]

    elems += [Spacer(1,0.3*cm), HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=6),
              Paragraph("Disease Information", SE)]
    for label, key in [('Description','description'),('Possible Causes','causes'),
                       ('Common Symptoms','symptoms'),('Recommended Treatment','treatment')]:
        elems += [Paragraph(f"<b>{label}:</b>", B), Paragraph(result.get(key,'N/A'), B), Spacer(1,0.1*cm)]

    elems += [Spacer(1,0.4*cm),
              HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=6),
              Paragraph("This report is AI-assisted and does NOT replace professional medical diagnosis. "
                        "Always consult a certified dermatologist. DermaScan AI — Educational use only.", D)]
    doc.build(elems)
    buffer.seek(0)
    return buffer

# ─── AUTH ROUTES ──────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form.get('name',     '').strip()
        email    = request.form.get('email',    '').strip().lower()
        password = request.form.get('password', '').strip()

        if not name or not email or not password:
            return render_template('register.html', error='All fields are required.', name=name, email=email)
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters.', name=name, email=email)
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered. Please login.', name=name, email=email)

        new_user = User(
            name       = name,
            email      = email,
            password   = generate_password_hash(password),
            created_at = datetime.now().strftime('%d %B %Y')
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login', success='Account created! Please login.'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email    = request.form.get('email',    '').strip().lower()
        password = request.form.get('password', '').strip()
        user     = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return render_template('login.html', error='Invalid email or password.', email=email)
        session['user_id']   = user.id
        session['user_name'] = user.name
        return redirect(url_for('index'))

    success = request.args.get('success', '')
    return render_template('login.html', success=success)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─── MAIN ROUTES ──────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html', user_name=session.get('user_name', ''))

@app.route('/history')
@login_required
def history():
    return render_template('history.html', user_name=session.get('user_name', ''))

@app.route('/progress')
@login_required
def progress():
    return render_template('progress.html', user_name=session.get('user_name', ''))

# ─── API ROUTES ───────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type.'}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'],  exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

    filename     = secure_filename(file.filename)
    save_path    = os.path.join(app.config['UPLOAD_FOLDER'],  filename)
    heatmap_name = f"heatmap_{filename}"
    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_name)
    file.save(save_path)

    try:
        gradcam = generate_gradcam(save_path, heatmap_path)  #先 gradcam ✅
        result  = predict_disease(save_path)                  # அப்புறம் predict ✅
        
        symptoms_text = request.form.get('symptoms', '')
        now           = datetime.now()

        record = ScanRecord(
            user_id           = session['user_id'],
            date              = now.strftime('%d %B %Y'),
            time              = now.strftime('%I:%M %p'),
            disease           = result['predicted_disease'],
            confidence        = result['confidence'],
            uncertainty       = result['uncertainty'],        # ← NEW saved to DB
            uncertainty_level = result['uncertainty_level'],  # ← NEW saved to DB
            severity          = result.get('severity', ''),
            symptoms          = symptoms_text,
            image_file        = filename,
            heatmap_file      = heatmap_name if gradcam else None,
        )
        db.session.add(record)
        db.session.commit()

        result['image_url']   = f"/static/uploads/{filename}"
        result['heatmap_url'] = f"/static/heatmaps/{heatmap_name}" if gradcam else None
        result['filename']    = filename
        result['record_id']   = record.id
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/symptom_check', methods=['POST'])
@login_required
def symptom_check():
    data = request.get_json()
    text = data.get('symptoms', '').strip()
    if not text:
        return jsonify({'error': 'No symptoms provided.'}), 400
    matches = match_symptoms_from_text(text)
    if not matches:
        return jsonify({'matched': False,
                        'message': 'No matching disease found. Upload an image for accurate diagnosis.',
                        'matches': []}), 200
    return jsonify({'matched': True, 'matches': matches,
                    'message': f"Found {len(matches)} possible condition(s)."}), 200


@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    records = ScanRecord.query.filter_by(
        user_id=session['user_id']
    ).order_by(ScanRecord.id.desc()).all()
    return jsonify([r.to_dict() for r in records]), 200


@app.route('/get_chart_data', methods=['GET'])
@login_required
def get_chart_data():
    records = ScanRecord.query.filter_by(
        user_id=session['user_id']
    ).order_by(ScanRecord.id.asc()).all()
    data = {}
    for r in records:
        if r.disease not in data:
            data[r.disease] = []
        data[r.disease].append({
            'date':            r.date,
            'confidence':      r.confidence,
            'uncertainty':     r.uncertainty,      # ← NEW in chart data
            'id':              r.id,
        })
    return jsonify(data), 200


@app.route('/delete_record/<int:record_id>', methods=['DELETE'])
@login_required
def delete_record(record_id):
    record = ScanRecord.query.filter_by(
        id=record_id, user_id=session['user_id']
    ).first_or_404()
    db.session.delete(record)
    db.session.commit()
    return jsonify({'message': 'Record deleted'}), 200


@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    result        = json.loads(request.form.get('result'))
    filename      = request.form.get('filename')
    symptoms_text = request.form.get('symptoms_text', '')
    img_path      = os.path.join(app.config['UPLOAD_FOLDER'],  filename)
    heatmap_path  = os.path.join(app.config['HEATMAP_FOLDER'], f"heatmap_{filename}")
    try:
        pdf         = generate_pdf_report(result, img_path, heatmap_path, filename, symptoms_text)
        report_name = f"DermaScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(pdf, mimetype='application/pdf',
                         as_attachment=True, download_name=report_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_episodes/<disease>', methods=['GET'])
@login_required
def get_episodes(disease):
    episodes = Episode.query.filter_by(
        user_id=session['user_id'],
        disease=disease,
        status='active'
    ).order_by(Episode.id.asc()).all()

    all_disease_eps = Episode.query.filter_by(
        user_id=session['user_id'],
        disease=disease
    ).order_by(Episode.id.asc()).all()

    id_to_num = {ep.id: i+1 for i, ep in enumerate(all_disease_eps)}
    result = []
    for ep in episodes:
        ep_dict = ep.to_dict()
        ep_dict['episode_num']  = id_to_num.get(ep.id, 1)
        ep_dict['display_name'] = f"{ep.disease} — Episode {id_to_num.get(ep.id, 1)}"
        result.append(ep_dict)
    return jsonify(result), 200


@app.route('/create_episode', methods=['POST'])
@login_required
def create_episode():
    data    = request.get_json()
    disease = data.get('disease', '').strip()
    notes   = data.get('notes', '').strip()
    if not disease:
        return jsonify({'error': 'Disease name is required.'}), 400
    now     = datetime.now()
    episode = Episode(
        user_id    = session['user_id'],
        disease    = disease,
        start_date = now.strftime('%d %B %Y'),
        status     = 'active',
        notes      = notes,
    )
    db.session.add(episode)
    db.session.commit()
    return jsonify({'episode': episode.to_dict(), 'message': 'Episode created!'}), 200


@app.route('/link_episode', methods=['POST'])
@login_required
def link_episode():
    data       = request.get_json()
    record_id  = data.get('record_id')
    episode_id = data.get('episode_id')
    record  = ScanRecord.query.filter_by(id=record_id,  user_id=session['user_id']).first_or_404()
    episode = Episode.query.filter_by(id=episode_id, user_id=session['user_id']).first_or_404()
    record.episode_id = episode_id
    db.session.commit()
    return jsonify({'message': 'Scan linked to episode!'}), 200


@app.route('/cure_episode/<int:episode_id>', methods=['POST'])
@login_required
def cure_episode(episode_id):
    episode          = Episode.query.filter_by(id=episode_id, user_id=session['user_id']).first_or_404()
    episode.status   = 'cured'
    episode.end_date = datetime.now().strftime('%d %B %Y')
    db.session.commit()
    return jsonify({'message': f'{episode.disease} episode marked as cured!'}), 200


@app.route('/get_all_episodes', methods=['GET'])
@login_required
def get_all_episodes():
    episodes = Episode.query.filter_by(
        user_id=session['user_id']
    ).order_by(Episode.id.asc()).all()

    disease_count = {}
    result        = []
    for ep in episodes:
        disease_count[ep.disease] = disease_count.get(ep.disease, 0) + 1
        ep_num = disease_count[ep.disease]
        scans  = ScanRecord.query.filter_by(episode_id=ep.id).order_by(ScanRecord.id.asc()).all()
        result.append({
            'id':           ep.id,
            'episode_num':  ep_num,
            'display_name': f"{ep.disease} — Episode {ep_num}",
            'disease':      ep.disease,
            'start_date':   ep.start_date,
            'end_date':     ep.end_date,
            'status':       ep.status,
            'notes':        ep.notes,
            'scan_count':   len(scans),
            'scans': [
                {
                    'id':          s.id,
                    'date':        s.date,
                    'confidence':  s.confidence,
                    'uncertainty': s.uncertainty,  # ← NEW in episode scans
                }
                for s in scans
            ]
        })

    result.reverse()
    return jsonify(result), 200


# ─── ENTRY POINT ──────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

    with app.app_context():
        db.create_all()
        print("[OK] Database ready")

   

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
