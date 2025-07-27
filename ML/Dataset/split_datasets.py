import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

def split_dataset(dataset_dir, prefix):
    # List all jpg files
    images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    test_images = images[:10]
    train_images = images[10:]

    # Create train and test directories
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move images
    for img in test_images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(test_dir, img))
    for img in train_images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(train_dir, img))
    print(f"{prefix}: {len(train_images)} train, {len(test_images)} test images.")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    split_dataset(os.path.join(base, '1_Crop_Disease_DS'), 'Crop Disease')
    split_dataset(os.path.join(base, '2_Crop_Insect_DS'), 'Crop Insect') 