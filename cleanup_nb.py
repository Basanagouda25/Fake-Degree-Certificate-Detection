import json
import os

nb_path = r"c:\Users\basan\Desktop\project\Minor_Project\Fake_certificate_detection.ipynb"

cell_1_code = """import os
import shutil

# 1. Use the exact ABSOLUTE path to your Minor_Project folder
project_dir = r"c:\Users\basan\Desktop\project\Minor_Project"

base_path = os.path.join(project_dir, "maindataset")
output_path = os.path.join(project_dir, "dataset_clean")

# Create folders
for split in ["train", "val"]:
    for cls in ["real", "fake"]:
        os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

def process_folder(exact_folder_path, dest_name):
    images_path = os.path.join(exact_folder_path, "images")
    labels_path = os.path.join(exact_folder_path, "labels")

    if not os.path.exists(labels_path):
        print(f"⚠️ WARNING: Directory not found: {labels_path}")
        return

    print(f"Processing: {labels_path} ...")
    for file in os.listdir(labels_path):
        if file.endswith(".txt"):
            label_file = os.path.join(labels_path, file)
            with open(label_file, "r") as f:
                first_line = f.readline().strip()
            if not first_line:
                continue

            class_id = int(first_line.split()[0])
            dest = "fake" if class_id == 0 else "real" if class_id == 1 else None
            
            image_name = file.replace(".txt", ".jpg")
            image_path = os.path.join(images_path, image_name)

            if dest and os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(output_path, dest_name, dest, image_name))

# 2. Process root "train" folder 
train_path = os.path.join(base_path, "train")
process_folder(train_path, "train")

# 3. Process the "valid" folder inside "d1Certificate forgery detection"
valid_path = os.path.join(base_path, "d1Certificate forgery detection", "valid")
process_folder(valid_path, "val")

print("✅ DONE! Dataset organized successfully.")"""

cell_2_code = """import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import RandomFlip, RandomRotation, Rescaling, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# 1. Use the EXACT ABSOLUTE PATH to your dataset_clean folder
DATASET_PATH = r"c:\Users\basan\Desktop\project\Minor_Project\dataset_clean" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def build_and_train_model():
    print("Loading training dataset...")
    train_dataset = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='binary'
    )

    print("Loading validation dataset...")
    val_dataset = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"),
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='binary'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),
    ], name="data_augmentation")

    print("Building MobileNetV2 model...")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = Rescaling(1./255)(x)
    x = base_model(x, training=False) 
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()

    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model_path = r"c:\Users\basan\Desktop\project\Minor_Project\fake_certificate_model.h5"
    model.save(model_path)
    print(f"Model saved successfully to '{model_path}'")
    return model_path

def predict_single_image(model_path, img_path):
    print(f"\\nEvaluating single image: {img_path}")
    if not os.path.exists(img_path):
        print("Error: Image path not found!")
        return

    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    print(f"Raw Prediction Probability: {score:.4f}")
    if score >= 0.5:
        print("Prediction result: Real Certificate")
    else:
        print("Prediction result: Fake Certificate")

# Uncomment to train
saved_model_path = build_and_train_model()
"""

# Open notebook
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Clear cells
nb["cells"] = []

def make_cell(code):
    lines = code.split("\n")
    formatted = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": formatted
    }

nb["cells"].append(make_cell(cell_1_code))
nb["cells"].append(make_cell(cell_2_code))

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook thoroughly cleaned and streamlined!")
