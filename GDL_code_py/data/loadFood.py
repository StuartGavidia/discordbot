import os
import random
from shutil import copyfile

food_101_dataset_path = '/path/to/food-101/images'

output_folder_path = '/food'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

categories = os.listdir(food_101_dataset_path)

selected_images = []
while len(selected_images) < 50000:
    category = random.choice(categories)
    category_path = os.path.join(food_101_dataset_path, category)
    image = random.choice(os.listdir(category_path))
    image_path = os.path.join(category_path, image)
    if image_path not in selected_images:
        selected_images.append(image_path)

for idx, image_path in enumerate(selected_images):
    new_image_path = os.path.join(output_folder_path, f'food_{idx}.jpg')
    copyfile(image_path, new_image_path)
    print(f'Copied {image_path} to {new_image_path}')

print('Image extraction and copying completed.')
