import os
import json
import numpy as np
import cv2
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH = "model/skin_disease_model.h5"
CLASS_JSON = "model/class_names.json"
IMG_SIZE   = 224

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ Model loaded!")

# ── Load class names from saved JSON ─────────────────────────────
if os.path.exists(CLASS_JSON):
    with open(CLASS_JSON) as f:
        classes = json.load(f)
    print(f"✓ Classes loaded: {classes}")
else:
    # fallback if JSON not found
    classes = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]
    print("⚠ class_names.json not found — using default order")

# ── Preprocess function ───────────────────────────────────────────
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found at: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ← KEY FIX: BGR to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)           # shape → (1, 224, 224, 3)
    return img

# ── Predict function ──────────────────────────────────────────────
def predict(image_path):
    img         = preprocess_image(image_path)
    predictions = model.predict(img, verbose=0)[0]  # shape → (5,)

    # Top prediction
    top_index   = np.argmax(predictions)
    top_disease = classes[top_index]
    top_conf    = predictions[top_index] * 100

    # All scores sorted
    all_scores = sorted(
        [{'disease': classes[i], 'confidence': round(float(predictions[i]) * 100, 2)}
         for i in range(len(classes))],
        key=lambda x: x['confidence'],
        reverse=True
    )

    return top_disease, top_conf, all_scores

# ── Test it ───────────────────────────────────────────────────────
image_path = "test_image.jpg"   # ← put your test image name here

try:
    disease, confidence, all_scores = predict(image_path)

    print("\n" + "=" * 40)
    print(f"  Predicted Disease : {disease}")
    print(f"  Confidence        : {confidence:.2f}%")
    print("=" * 40)
    print("\n  All scores:")
    for s in all_scores:
        bar = "█" * int(s['confidence'] / 5)
        print(f"  {s['disease']:<12} {s['confidence']:>6.2f}%  {bar}")
    print("=" * 40)

except ValueError as e:
    print(f"Error: {e}")