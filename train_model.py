import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

print("=" * 50)
print("   DermaScan AI — Phase 1 Only (Saves as .h5)")
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
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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

# ── Save class names ──────────────────────────────────────────────
class_indices = train_data.class_indices
class_names   = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
print(f"\n  Class order  : {class_names}")
print(f"  Train images : {train_data.samples}")
print(f"  Test images  : {test_data.samples}")

os.makedirs('model', exist_ok=True)
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# ── Build Model ───────────────────────────────────────────────────
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze all — no fine-tuning

x      = base_model.output
x      = tf.keras.layers.GlobalAveragePooling2D()(x)
x      = tf.keras.layers.Dense(256, activation='relu')(x)
x      = tf.keras.layers.Dropout(0.4)(x)
x      = tf.keras.layers.Dense(128, activation='relu')(x)
x      = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n  Model built — {len(model.layers)} layers")
print(f"  Base model frozen — training top layers only")

# ── Callbacks — manually save best model to avoid .h5 corruption ─
# We use EarlyStopping with restore_best_weights=True
# Then manually save ONCE at the end — this avoids the deepcopy crash
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,  # keeps best weights in memory
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ── Train ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   Training (25 epochs max)")
print("=" * 50 + "\n")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=25,
    callbacks=callbacks,
    verbose=1
)

# ── Save ONCE at the end — avoids deepcopy crash with .h5 ─────────
print("\n  Saving model as .h5 ...")
model.save("model/skin_disease_model.h5", save_format='h5')
print("  Model saved to model/skin_disease_model.h5")

# ── Final Evaluation ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("   Final Evaluation")
print("=" * 50)

best_model = tf.keras.models.load_model("model/skin_disease_model.h5")
loss, accuracy = best_model.evaluate(test_data, verbose=0)

print(f"\n  Test Accuracy  : {accuracy * 100:.2f}%")
print(f"  Test Loss      : {loss:.4f}")
print(f"\n  Model saved    → model/skin_disease_model.h5")
print(f"  Classes saved  → model/class_names.json")
print(f"\n  CLASS_NAMES for app.py : {class_names}")

if accuracy * 100 >= 80:
    print("\n  Excellent! Ready to build Grad-CAM + PDF features.")
elif accuracy * 100 >= 75:
    print("\n  Good accuracy. Proceed to build features.")
else:
    print("\n  Accuracy low — check dataset quality.")

print("\n" + "=" * 50)