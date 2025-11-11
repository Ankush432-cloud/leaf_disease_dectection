import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import numpy as np

# Set paths
train_dir = 'Datasets/train'
val_dir = 'Datasets/valid'
model_save_path = 'model/leaf_disease_model1.h5'
plot_save_path = 'plots/training_accuracy.png'

# Set image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 40
CONFIDENCE_THRESHOLD = 0.6  # below this, return Unknown

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Robust data augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,  # Plants don't grow upside down
    shear_range=0.2,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest',
    validation_split=0.2  # Internal validation split
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Get dataset information
num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())

print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_data.filenames)}")
print(f"Validation samples: {len(val_data.filenames)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# Save class names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
with open('inference_config.json', 'w') as f:
    json.dump({"confidence_threshold": CONFIDENCE_THRESHOLD}, f, indent=2)

# Build robust ResNet50V2 model with transfer learning
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Add robust classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)  # High dropout to prevent overfitting
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Use a lower learning rate to prevent overfitting
optimizer = Adam(learning_rate=0.0001)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define robust callbacks
checkpoint = ModelCheckpoint(
    model_save_path, 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=12,  # More patience
    restore_best_weights=True, 
    verbose=1
)
lr_reduce = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.3,  # Less aggressive reduction
    patience=6, 
    verbose=1,
    min_lr=1e-7
)

print("Starting robust training with ResNet50V2 transfer learning...")
print("Model Summary:")
model.summary()

# Phase 1: Train with frozen base model
print("\n=== PHASE 1: Training with frozen base model ===")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[checkpoint, early_stop, lr_reduce],
    verbose=1
)

# Phase 2: Fine-tune the base model
print("\n=== PHASE 2: Fine-tuning base model ===")
# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

# Recompile with lower learning rate
optimizer = Adam(learning_rate=0.00001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[checkpoint, early_stop, lr_reduce],
    verbose=1
)

# Combine histories
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

# Plot training results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='s')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot learning rate
plt.subplot(1, 3, 3)
epochs = range(1, len(history['accuracy']) + 1)
plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.axvline(x=20, color='g', linestyle='--', label='Phase 2 Start')
plt.title('Training Progress with Phase Transition')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')

print(f"✅ Model training complete and saved as: {model_save_path}")
print(f"📈 Training plots saved at: {plot_save_path}")

# Print final metrics
final_train_acc = history['accuracy'][-1]
final_val_acc = history['val_accuracy'][-1]
print(f"🎯 Final Training Accuracy: {final_train_acc:.4f}")
print(f"🎯 Final Validation Accuracy: {final_val_acc:.4f}")

# Test model on validation set
print("\n=== MODEL VALIDATION ===")
val_predictions = model.predict(val_data, verbose=0)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_data.classes
val_max_conf = np.max(val_predictions, axis=1)
val_unknown_flags = val_max_conf < CONFIDENCE_THRESHOLD
print(f"Unknown (by threshold) count: {val_unknown_flags.sum()} of {len(val_max_conf)} at threshold {CONFIDENCE_THRESHOLD}")

# Calculate per-class accuracy
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(val_true_classes, val_pred_classes, target_names=class_names))

# Check for overfitting
train_val_diff = final_train_acc - final_val_acc
if train_val_diff > 0.1:
    print(f"⚠️ WARNING: Potential overfitting detected (train-val diff: {train_val_diff:.3f})")
else:
    print(f"✅ Good generalization (train-val diff: {train_val_diff:.3f})") 

# Simple single-image inference with Unknown rejection
def predict_image_with_unknown(img_path: str, threshold: float = CONFIDENCE_THRESHOLD):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]
    max_idx = int(np.argmax(probs))
    max_conf = float(np.max(probs))
    if max_conf < threshold:
        return {"label": "Unknown", "confidence": max_conf}
    return {"label": class_names[max_idx], "confidence": max_conf}

# Example:
# result = predict_image_with_unknown('path/to/leaf.jpg')
# print(result)

# Lightweight pre-filter to reject non-leaf images using green pixel ratio in HSV
def is_probably_leaf_image(img_path: str,
                           min_green_ratio: float = 0.15,
                           hsv_green_range: tuple = (0.20, 0.45),
                           min_saturation: float = 0.25,
                           min_value: float = 0.2) -> dict:
    """Returns a dict with fields: is_leaf (bool), green_ratio (float).
    Heuristic: an image dominated by leaf-like greens in HSV.
    """
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img) / 255.0  # [0,1]
    # Convert to HSV
    hsv = tf.image.rgb_to_hsv(tf.convert_to_tensor(x))
    h = hsv[..., 0].numpy()
    s = hsv[..., 1].numpy()
    v = hsv[..., 2].numpy()

    green_mask = (
        (h >= hsv_green_range[0]) & (h <= hsv_green_range[1]) &
        (s >= min_saturation) & (v >= min_value)
    )
    green_ratio = float(np.mean(green_mask))
    return {"is_leaf": green_ratio >= min_green_ratio, "green_ratio": green_ratio}

def predict_image_safe(img_path: str,
                       threshold: float = CONFIDENCE_THRESHOLD,
                       min_green_ratio: float = 0.15,
                       min_sharpness: float = 12.0) -> dict:
    """Pre-filter with leaf heuristic, then classify with Unknown rejection.
    Returns dict: {label, confidence, green_ratio, sharpness}
    """
    leaf_check = is_probably_leaf_image(img_path, min_green_ratio=min_green_ratio)
    if not leaf_check["is_leaf"]:
        return {"label": "Not a leaf", "confidence": 0.0, "green_ratio": leaf_check["green_ratio"], "sharpness": 0.0}
    sharp_check = is_sharp_enough(img_path, min_sharpness=min_sharpness)
    if not sharp_check["is_sharp"]:
        return {"label": "Image too blurry", "confidence": 0.0, "green_ratio": leaf_check["green_ratio"], "sharpness": sharp_check["sharpness"]}
    result = predict_image_with_unknown(img_path, threshold=threshold)
    result["green_ratio"] = leaf_check["green_ratio"]
    result["sharpness"] = sharp_check["sharpness"]
    return result

# Example safe prediction:
# result = predict_image_safe('path/to/any_image.jpg')
# print(result)

# Sharpness/blur detector using Sobel gradients (Tenengrad-like score)
def compute_sharpness_score(img_path: str) -> float:
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img) / 255.0  # [H,W,3]
    # Convert to grayscale using standard luminance coefficients
    gray = 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]
    gray = tf.convert_to_tensor(gray[None, ..., None], dtype=tf.float32)  # [1,H,W,1]
    sobel = tf.image.sobel_edges(gray)  # [1,H,W,1,2]
    gx = sobel[..., 0]
    gy = sobel[..., 1]
    grad_mag = tf.sqrt(gx * gx + gy * gy)  # [1,H,W,1]
    # Tenengrad score ~ mean of gradient magnitude squared or magnitude
    score = tf.reduce_mean(grad_mag * grad_mag)
    return float(score.numpy())

def is_sharp_enough(img_path: str, min_sharpness: float = 12.0) -> dict:
    score = compute_sharpness_score(img_path)
    return {"is_sharp": score >= min_sharpness, "sharpness": score}