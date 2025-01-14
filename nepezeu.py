import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# FUNCTION TO LOAD AND PROCESS IMAGES
def load_images_from_folder(folder_path, img_size=(200, 200)):
    images = []
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading images from {folder_path}"):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert("L")  # CONVERT TO GRAYSCALE
            img = img.resize(img_size)  # RESIZE TO 200x200
            images.append(np.array(img))  # CONVERT TO NUMPY ARRAY
        except Exception as e:
            print(f"Could not process {filename}: {e}")
    return np.array(images)

# LOAD IMAGES FROM BOTH FOLDERS
emotion_images = load_images_from_folder("task-emotion")
rest_images = load_images_from_folder("task-rest")

# SAVE TO .NPZ FILES
np.savez("task-emotion.npz", arr_0=emotion_images)
np.savez("task-rest.npz", arr_0=rest_images)

print("Datasets saved as task-emotion.npz and task-rest.npz")

