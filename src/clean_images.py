import os
from PIL import Image

DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

def clean_directory(root_dir):
    deleted_count = 0
    skipped_count = 0
    print(f"--- Checking directory: {root_dir} ---")
    
    # Iterate through all files in all subdirectories (classes)
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Only look at common image file extensions
            if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                # Open the image using Pillow
                img = Image.open(file_path)
                
                # Check for images that are not standard RGB (mode 'L' is grayscale).
                if img.mode != 'RGB':
                    os.remove(file_path)
                    deleted_count += 1

            except PermissionError:
                print(f"   ⚠️ Skipping locked file: {file_path}")
                skipped_count += 1
                continue # Skip this file and move to the next

            except Exception as e:
                # Catch corrupted files that PIL can't even open
                print(f"   ❌ Deleting corrupted file: {file_path}")
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except PermissionError:
                    print(f"   ⚠️ Skipping locked file (corrupted but locked): {file_path}")
                    skipped_count += 1
                
    return deleted_count, skipped_count

total_deleted = 0
total_skipped = 0

deleted_train, skipped_train = clean_directory(TRAIN_DIR)
total_deleted += deleted_train
total_skipped += skipped_train

deleted_test, skipped_test = clean_directory(TEST_DIR)
total_deleted += deleted_test
total_skipped += skipped_test

print(f"\n✅ Image cleanup complete!")
print(f"   Files deleted: {total_deleted}")
print(f"   Files locked/skipped: {total_skipped}. These must be manually removed.")