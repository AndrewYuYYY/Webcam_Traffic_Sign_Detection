import os
import random
import shutil

# Paths to your dataset
images_path = "dataset/images"
labels_path = "dataset/labels"
train_images_path = "dataset_split/train/images"
train_labels_path = "dataset_split/train/labels"
val_images_path = "dataset_split/val/images"
val_labels_path = "dataset_split/val/labels"

# Create directories for train and validation splits
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

# Ensure randomization of the dataset
random.shuffle(image_files)

# Split the dataset
split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]


# Copy files to their respective directories
def copy_files(files, src_images_path, src_labels_path, dest_images_path, dest_labels_path):
    for image_file in files:
        # Define source and destination paths for images
        src_image = os.path.join(src_images_path, image_file)
        dest_image = os.path.join(dest_images_path, image_file)

        # Copy the image file
        shutil.copy(src_image, dest_image)

        # Define source and destination paths for labels
        label_file = os.path.splitext(image_file)[0] + ".txt"  # Match the label file
        src_label = os.path.join(src_labels_path, label_file)
        dest_label = os.path.join(dest_labels_path, label_file)

        # Check if the label file exists before copying
        if os.path.exists(src_label):
            shutil.copy(src_label, dest_label)
        else:
            print(f"Warning: Label file not found for {image_file}")


# Copy the training and validation files
copy_files(train_files, images_path, labels_path, train_images_path, train_labels_path)
copy_files(val_files, images_path, labels_path, val_images_path, val_labels_path)

print(f"Dataset split completed! Training images: {len(train_files)}, Validation images: {len(val_files)}")