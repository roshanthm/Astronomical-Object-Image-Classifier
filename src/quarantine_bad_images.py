import os
from PIL import Image
import shutil

DATA_DIR = "data/train"
TEST_DIR = "data/test"
BAD_DIR = "data/bad_images"

os.makedirs(BAD_DIR, exist_ok=True)

def check_and_move(path):
    try:
        img = Image.open(path)
        img.verify()  # detect corruption
        img = Image.open(path).convert("RGB")

        if img.mode != "RGB":
            raise ValueError("Not RGB")

    except Exception as e:
        print(f"[BAD] {path} --> {e}")
        # move corrupted file
        dst = os.path.join(BAD_DIR, os.path.basename(path))
        shutil.move(path, dst)
        return False

    return True


def process_folder(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full = os.path.join(root, f)
                check_and_move(full)


print("Scanning TRAIN...")
process_folder(DATA_DIR)

print("Scanning TEST...")
process_folder(TEST_DIR)

print("\nDone. Bad images moved to:", BAD_DIR)
