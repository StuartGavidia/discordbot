import numpy as np
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import save_img

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CIFAR-10 labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# The 'horse' class is labeled as 7

# Filter out the 'horse' images
horse_train = x_train[y_train.flatten() == 7]
horse_test = x_test[y_test.flatten() == 7]

# Combine train and test datasets
horse_images = np.concatenate((horse_train, horse_test))

# Create a directory to save horse images
horse_dir = 'horse'
if not os.path.exists(horse_dir):
    os.makedirs(horse_dir)

# Save the horse images
for i, img in enumerate(horse_images):
    img_file = os.path.join(horse_dir, f'horse_{i}.png')
    save_img(img_file, img)

print(f'Horse images saved in {horse_dir}')
