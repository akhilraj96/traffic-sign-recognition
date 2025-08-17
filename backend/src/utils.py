import os
import sys
import pickle
import numpy as np
from PIL import Image


from src.exception import CustomException
from src.logger import logging

DATASET_PATH = "data"  # Example: backend/data/

IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 43

VALID_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif',
    '.tif', '.tiff', '.ppm', '.pgm', '.pbm', '.pnm',
    '.webp', '.ico', '.icns', '.tga', '.pcx',
    '.dds', '.heif', '.heic'
}

def load_data(data_path):
    try:
        images = []
        labels = []

        for class_id in range(NUM_CLASSES):
            class_path = os.path.join(data_path, str(class_id))
            if not os.path.exists(class_path):
                print(f"⚠ Warning: Missing folder for class {class_id}")
                continue

            for img_file in os.listdir(class_path):
                _, ext = os.path.splitext(img_file)
                if ext.lower() not in VALID_EXTENSIONS:
                    continue  # Skip non-image files

                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    images.append(np.array(img))
                    labels.append(class_id)
                except Exception as e:
                    print(f"❌ Error loading image '{img_file}' in class {class_id}: {e}")

        return np.array(images), np.array(labels)

    except Exception as e:
        raise CustomException(e, sys)

def load_data_2(file):
	try:
		with open(file, mode='rb') as f:
			file_ = pickle.load(f)
		x_, y_ = file_['features'], file_['labels']
		logging.info(file+" Loaded")
		return x_, y_
	except Exception as e:
		raise CustomException(e, sys)

def save_data(data_X, data_y, file_path):
	try:
		# Ensure the directory exists
		dir_path = os.path.dirname(file_path)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

		data = {
			'features': data_X,
			'labels': data_y
		}

		with open(file_path, 'wb') as file:
			pickle.dump(data, file)

		logging.info(f'data saved to {file_path}')
	except Exception as e:
		raise CustomException(e, sys)



def load_object(file_path):
	try:
		with open(file_path, 'rb') as f:
			model = pickle.load(f)
		return model
	except Exception as e:
		raise CustomException(e, sys)
