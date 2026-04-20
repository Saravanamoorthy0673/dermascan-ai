import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("=" * 50)
print("   Phase 2 Fix — Fine-tuning from best weights")
print("=" * 50)

IMG_SIZE   = 224
BATCH_SIZE = 32
train_dir  = "Dataset/train"
test_dir   = "Dataset/test"

# ── Data generators ───────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.25,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ── Load best model from Phase 1 ─────────────────────────────────
print("\n✓ Loading best model from Phase 1...")
model = load_model("model/skin_disease_model.h5")

# Check current accuracy
loss, acc = model.evaluate(test_data, verbose=0)
print(f"✓ Current saved model accuracy: {acc*100:.2f}%")

# ── Find EfficientNetB0 base model layer automatically ────────────
print("\n✓ Scanning model layers...")
base_model = None
for i, layer in enumerate(model.layers):
    print(f"  Layer {i}: {layer.name} — {type(layer).__name__}")
    if 'efficientnet' in layer.name.lower():
        base_model = layer
        print(f"  → Found EfficientNetB0 at layer {i}")
        break

if base_model is None:
    print("\n⚠ EfficientNetB0 not found as sublayer.")
    print("  Trying to unfreeze full model top layers instead...")

    # Fallback: unfreeze last 50 layers of the full model directly
    for layer in model.layers[-50:]:
        layer.trainable = True
    print(f"  Unfroze last 50 layers of full model")
else:
    # Unfreeze only last 20 layers of EfficientNetB0
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    trainable = sum([1 for l in base_model.layers if l.trainable])
    print(f"✓ Unfrozen {trainable} layers in EfficientNetB0")

# ── Compile with very low learning rate ──────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "model/skin_disease_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
]

# ── Fine-tune ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   Fine-tuning (15 epochs max)")
print("=" * 50)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# ── Final result ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   Final Result")
print("=" * 50)

best_model = load_model("model/skin_disease_model.h5")
loss, accuracy = best_model.evaluate(test_data, verbose=0)

print(f"\n  Final Test Accuracy : {accuracy * 100:.2f}%")
print(f"  Final Test Loss     : {loss:.4f}")

if accuracy * 100 >= 82:
    print("\n  ✅ Excellent! Ready to build Grad-CAM + PDF features.")
elif accuracy * 100 >= 75:
    print("\n  ✅ Good — acceptable for final year project.")
else:
    print("\n  ⚠ Accuracy still low.")

print("\n" + "=" * 50)