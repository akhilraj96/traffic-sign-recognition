import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to dataset folder
DATASET_PATH = "data"
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 43  # GTSRB dataset classes

def load_data(data_path):
    images = []
    labels = []

    for class_id in range(NUM_CLASSES):
        class_path = os.path.join(data_path, str(class_id))
        if not os.path.exists(class_path):
            continue
        for img_file in os.listdir(class_path)[:1]:  # limit to first 5 for speed
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            images.append(np.array(img))
            labels.append(class_id)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    X, y = load_data(DATASET_PATH)

    print(f"Dataset loaded: {X.shape[0]} images, {len(set(y))} unique labels")

    # Show first 20 images
    plt.figure(figsize=(10, 5))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(X[i])
        plt.title(f"Class: {y[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
