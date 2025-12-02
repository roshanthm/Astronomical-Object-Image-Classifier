# notebooks/train_model.py
# Robust trainer: safe TF-only image loader (handles 1,3,4-channel images)
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# --- Config ---
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root when run from notebooks/
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# --- Utilities: discover class names from folders ---
def get_class_names(folder):
    classes = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
    return classes

class_names = get_class_names(TRAIN_DIR)
num_classes = len(class_names)
print("Classes:", class_names)

# Map class name -> index
class_to_index = {name: idx for idx, name in enumerate(class_names)}

# --- Build dataset of (path, label) pairs ---
def paths_and_labels_from_dir(data_dir):
    paths = []
    labels = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_index[cls])
    return paths, labels

def tf_dataset_from_paths(paths, labels, shuffle=True):
    paths = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    return ds

# --- Robust decode & normalize pipeline (TF ops only) ---
def decode_and_process(path, label):
    # path is a string tensor (path)
    image_bytes = tf.io.read_file(path)
    # decode image without forcing channels (channels=0). expand_animations=False to avoid GIF issues.
    img = tf.image.decode_image(image_bytes, channels=0, expand_animations=False)
    img.set_shape([None, None, None])  # unknown H, W, C

    # Ensure we have a numeric channel dimension
    shape = tf.shape(img)
    channels = shape[-1]

    # Convert to float32 in range [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # If grayscale (1) -> convert to RGB
    def to_rgb():
        return tf.image.grayscale_to_rgb(img)

    # If RGBA (4) -> drop alpha channel
    def drop_alpha():
        return img[..., :3]

    def identity():
        return img

    img = tf.switch_case(
        tf.cast(channels, tf.int32),
        branch_fns={
            1: to_rgb,
            3: identity,
            4: drop_alpha
        },
        default=identity
    )

    # Final safety: if channels not in {1,3,4} (weird), return a zero-size tensor flagged for filtering
    final_shape = tf.shape(img)
    valid = tf.logical_and(final_shape[-1] == 3, final_shape[0] > 0)

    # Resize and normalize to IMG_SIZE
    img = tf.image.resize(img, IMG_SIZE)
    return img, label, valid

# Wrap decode to be used in map (we will filter invalid)
def map_decode(path, label):
    img, label, valid = tf.py_function(lambda p, l: decode_and_process_tf(p, l),
                                       inp=[path, label],
                                       Tout=[tf.float32, tf.int32, tf.bool])
    # py_function loses shape inference; set static shapes
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    label.set_shape([])
    valid.set_shape([])
    return img, label, valid

# We need a pure-TF wrapper callable because tf.py_function expects numpy/TF interplay;
# but simpler/cleaner approach: implement decode_and_process using TF ops directly (no nested py_function).
def decode_and_process_tf(path, label):
    # This function is intended to be called inside tf.py_function with numpy inputs,
    # but to avoid complications, we'll implement a small python-level safe loader.
    # However, to keep full TF-graph compatibility we will instead implement using tf ops below
    # (so override and use a simpler pipeline): actually we will not call this function directly.
    raise RuntimeError("Not used directly.")

# Simpler TF-only map implementation (no py_function) -- recommended:
def parse_tf(path, label):
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(image_bytes, channels=0, expand_animations=False)
    img.set_shape([None, None, None])
    img = tf.image.convert_image_dtype(img, tf.float32)
    ch = tf.shape(img)[-1]

    # handle channels cases
    img = tf.cond(tf.equal(ch, 1),
                  lambda: tf.image.grayscale_to_rgb(img),
                  lambda: tf.cond(tf.equal(ch, 4),
                                  lambda: img[..., :3],
                                  lambda: img))
    # verify final channels == 3
    final_ch = tf.shape(img)[-1]
    valid = tf.equal(final_ch, 3)

    img = tf.image.resize(img, IMG_SIZE)
    return img, label, valid

# Build final dataset: map(parse_tf) then filter on valid
def build_final_ds(paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf_dataset_from_paths(paths, labels, shuffle=shuffle)
    ds = ds.map(lambda p, l: parse_tf(p, l), num_parallel_calls=AUTOTUNE)
    # filter invalid
    ds = ds.filter(lambda img, label, valid: valid)
    # drop the valid flag
    ds = ds.map(lambda img, label, valid: (img, tf.one_hot(label, num_classes)), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# --- Create datasets ---
train_paths, train_labels = paths_and_labels_from_dir(TRAIN_DIR)
val_paths, val_labels = paths_and_labels_from_dir(TEST_DIR)

print(f"Found {len(train_paths)} training files and {len(val_paths)} validation files across {num_classes} classes.")

train_ds = build_final_ds(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
val_ds = build_final_ds(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

# --- Build model (Transfer Learning MobileNetV2) ---
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- Train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save model
os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)
model.save(os.path.join(ROOT, "model", "astronomical_classifier.keras"))
print("Training complete and model saved.")
