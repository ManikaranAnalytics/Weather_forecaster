"""
train.py  —  Improved Cloud Classification Trainer
MobileNetV2 with 2-phase fine-tuning, data augmentation, and callbacks.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ─── CONFIG ────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_FT   = 10     # Phase 1: top layers only (frozen base)
EPOCHS_FN   = 5      # Phase 2: fine-tune last 30 base layers
NUM_CLASSES = 7
DATASET_DIR = "dataset"
MODEL_PATH  = "cloud_model.keras"

CLASS_NAMES = [
    "Cumulus", "Altocumulus", "Cirrus",
    "ClearSky", "Stratocumulus", "Cumulonimbus", "Mixed"
]

# ─── DATA GENERATORS ────────────────────────────────────────────
# Training: augmentation + rescale
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
)
# Validation: only rescale (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)
val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
)

# ─── MODEL ARCHITECTURE ─────────────────────────────────────────
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False   # Freeze in Phase 1

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

# ─── CALLBACKS ──────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', patience=4,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=2, min_lr=1e-7, verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

# ─── PHASE 1: Train top layers only (base frozen) ───────────────
print("\n" + "="*50)
print("Phase 1: Training top layers (base frozen)")
print("="*50)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FT,
    callbacks=callbacks,
)

# ─── PHASE 2: Fine-tune last 30 layers of base ──────────────────
print("\n" + "="*50)
print("Phase 2: Fine-tuning last 30 base layers")
print("="*50)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

# Lower LR to avoid destroying pretrained weights
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FN,
    callbacks=callbacks,
)

# ─── FINAL SAVE & REPORT ────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\n✅ Model saved → {MODEL_PATH}")

val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"Final Validation Accuracy : {val_acc * 100:.2f}%")
print(f"Final Validation Loss     : {val_loss:.4f}")