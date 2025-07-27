import os
import shutil

# Paths
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
labels_train_dir = os.path.join(base, 'labels', 'train')
labels_test_dir = os.path.join(base, 'labels', 'test')
images_test_dir = os.path.join(base, 'images', 'test')

# Create test labels directory if it doesn't exist
os.makedirs(labels_test_dir, exist_ok=True)

# Get test image basenames (without extension)
test_images = [f for f in os.listdir(images_test_dir) if f.endswith('.jpg')]
test_basenames = [os.path.splitext(f)[0] for f in test_images]

# Move corresponding label files
for basename in test_basenames:
    label_file = f"{basename}.txt"
    src = os.path.join(labels_train_dir, label_file)
    dst = os.path.join(labels_test_dir, label_file)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {label_file} to test labels.")
    else:
        print(f"Label file {label_file} not found in train labels.") 