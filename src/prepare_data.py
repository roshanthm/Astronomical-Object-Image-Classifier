# src/prepare_data.py
import os
import sys

ROOT = 'data'
TRAIN = os.path.join(ROOT, 'train')
TEST  = os.path.join(ROOT, 'test')

print("Starting data preparation check...")

if os.path.exists(TRAIN) and os.path.exists(TEST):
    print(f"Detected existing train/test folders:\n - {TRAIN}\n - {TEST}")
    # Print a quick summary
    def summarize(folder):
        classes = []
        total = 0
        if not os.path.exists(folder):
            return classes, total
        for d in sorted(os.listdir(folder)):
            p = os.path.join(folder, d)
            if os.path.isdir(p):
                n = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                classes.append((d, n))
                total += n
        return classes, total

    tclasses, ttotal = summarize(TRAIN)
    vclasses, vtotal = summarize(TEST)
    print("\nTrain class counts:")
    for c,n in tclasses:
        print(f"  {c}: {n}")
    print(f"Total train images: {ttotal}")
    print("\nTest class counts:")
    for c,n in vclasses:
        print(f"  {c}: {n}")
    print(f"Total test images: {vtotal}")

    print("\nNo unzipping needed. Data structure looks OK.")
    sys.exit(0)

# ELSE: existing behavior if zip present (keeps your original logic minimal)
ZIP = os.path.join(ROOT, 'space-images-category.zip')
if not os.path.exists(ZIP):
    print(f"No zip found at {ZIP}. Place your source ZIP in data/ or arrange files manually.")
    sys.exit(1)
else:
    # If you want to re-extract, keep your previous extraction logic here
    print("Found ZIP -- extracting and preparing (not implemented automatically to avoid accidental deletes).")
    sys.exit(0)
