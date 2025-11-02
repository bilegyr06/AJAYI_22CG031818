import pandas as pd
import numpy as np
import tensorflow as tf  # NEW: Import for GPU/CPU context (ties to ML subfield, Lecture 1b)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # NEW: For imbalance (utility-based, Lecture 2b)
import matplotlib.pyplot as plt
import os

# Ensure data folder exists (structured env, Lecture 2a)
os.makedirs('data', exist_ok=True)

# Load FER-2013 dataset from data/ (knowledge representation: factored CSV, Lecture 2a)
try:
    df = pd.read_csv('data/fer2013.csv')
    print(f"Dataset loaded: {df.shape[0]} samples across 7 emotions")
except FileNotFoundError:
    print("Error: 'data/fer2013.csv' not found. Download from Kaggle and place in data/.")
    exit(1)  # Graceful exit (adaptability, Lecture 2b)

# Preprocess: Convert pixel strings to 48x48 grayscale arrays (normalize 0-1)
def preprocess_pixels(pixels_str):
    pixels = np.fromstring(pixels_str, dtype=int, sep=' ').reshape(48, 48)
    return pixels.astype(np.float32) / 255.0  # Rational scaling for CNN input

X = np.array([preprocess_pixels(p) for p in df['pixels']])  # Features: images
y = to_categorical(df['emotion'], num_classes=7)  # Labels: one-hot for 7 classes (Angry=0 to Neutral=6)

# Split: 80/20 train/test (supervised learning, Lecture 1b subfields)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['emotion'])

# Reshape for CNN (add channel dim: grayscale=1)
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# NEW: Class weights for imbalance (e.g., disgust underrepresented – fairness in learning agents, Lecture 2b)
class_weights = compute_class_weight('balanced', classes=np.unique(df['emotion']), y=df['emotion'])
class_weight_dict = {i: class_weights[i] for i in range(7)}
print("Class weights applied for balanced training.")

# Build model: Simple CNN (goal-based agent: goal=high accuracy; heuristics via filters)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # Edge detection (computer vision tie-in, Lecture 1b)
    MaxPooling2D(2, 2),  # Downsample (trade-off space/time, outcome iv)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting (adaptability, Lecture 2b)
    Dense(7, activation='softmax')  # Output: emotion probabilities
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  # Print architecture (agent structure, Lecture 2b)

# Train: Use GPU if available (your 1050 Ti for faster episodes) – FIXED: tf now imported
with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Tune for optimality (Lecture 2a: omniscience/learning)
        batch_size=64,  # Balance speed/memory
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,  # NEW: Apply weights
        verbose=1
    )

# Evaluate & Save (performance measure: rationality via metrics)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc:.4f} (CSC415 Goal: >0.60 for rational agent)")

model.save('face_emotionModel.h5')
print("Model saved as face_emotionModel.h5 – Ready for app.py integration!")

# Plot training history (visualize: train vs. val accuracy, outcome v: search algo analysis)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')  # For submission: ties to AI characteristics (Lecture 1b)
plt.show()