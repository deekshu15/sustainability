
import os
import csv
import shutil
from PIL import Image
import numpy as np
import traceback

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
REVIEW_DIR = os.path.join(DATA_ROOT, 'review', 'irrelevant')
CSV_PATH = os.path.join(DATA_ROOT, 'data_manifest.csv')

# Keywords to consider "irrelevant" for traffic analysis (mostly animals and non-vehicle fauna)
IRRELEVANT_KEYWORDS = {
    'dog','cat','bird','horse','cow','sheep','elephant','bear','zebra','giraffe',
    'lion','tiger','monkey','ape','kangaroo','otter','fox','wolf','pig','deer',
    'rabbit','rodent','hamster','squirrel','rat','mouse','insect','butterfly',
    'bee','ant','spider','snake','lizard','fish','shark','whale','dolphin','seal',
    'frog','toad','crab','lobster','shell','coral'
}

IMG_SIZE = (224, 224)


def ensure_dirs():
    os.makedirs(REVIEW_DIR, exist_ok=True)


def is_image_file(fname):
    ext = os.path.splitext(fname)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def pil_load_and_size(path):
    with Image.open(path) as im:
        im_rgb = im.convert('RGB')
        return im_rgb, im_rgb.size  # (width, height)


def prepare_for_model(pil_img):
    img = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def main():
    ensure_dirs()

    model = mobilenet_v2.MobileNetV2(weights='imagenet')

    rows = []
    total = 0
    flagged_count = 0
    errors = 0

    for dirpath, dirs, files in os.walk(DATA_ROOT):
        # skip the review folder if it exists inside data/
        if os.path.commonpath([os.path.abspath(dirpath), os.path.abspath(REVIEW_DIR)]) == os.path.abspath(REVIEW_DIR):
            continue
        # also skip created csv file
        if os.path.abspath(dirpath) == os.path.abspath(DATA_ROOT):
            # allow files in DATA_ROOT but skip the manifest if present
            pass

        for fname in files:
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, DATA_ROOT)
            if os.path.abspath(fpath) == os.path.abspath(CSV_PATH):
                continue
            if not is_image_file(fname):
                # record non-image files but skip prediction
                try:
                    size = os.path.getsize(fpath)
                except Exception:
                    size = -1
                rows.append([rel, os.path.basename(dirpath), size, '', '', '', '', '', '', '', 'NON_IMAGE', 'non-image'])
                continue

            total += 1
            try:
                pil_img, (w, h) = pil_load_and_size(fpath)
                size = os.path.getsize(fpath)
                arr = prepare_for_model(pil_img)
                preds = model.predict(arr)
                decoded = decode_predictions(preds, top=3)[0]
                # decoded is list of tuples (imagenet_id, label, prob)
                top_labels = [lab for (_, lab, prob) in decoded]
                top_probs = [float(prob) for (_, lab, prob) in decoded]
                # check for irrelevant keywords
                reasons = []
                for lab, prob in zip(top_labels, top_probs):
                    ll = lab.lower()
                    for kw in IRRELEVANT_KEYWORDS:
                        if kw in ll:
                            reasons.append(f"{lab}({prob:.2f})")
                            break
                flagged = len(reasons) > 0
                reason_str = ';'.join(reasons)

                rows.append([
                    rel,
                    os.path.basename(dirpath),
                    size,
                    w,
                    h,
                    top_labels[0] if len(top_labels) > 0 else '',
                    top_probs[0] if len(top_probs) > 0 else '',
                    top_labels[1] if len(top_labels) > 1 else '',
                    top_probs[1] if len(top_probs) > 1 else '',
                    top_labels[2] if len(top_labels) > 2 else '',
                    top_probs[2] if len(top_probs) > 2 else '',
                    'YES' if flagged else 'NO',
                    reason_str,
                ])

                if flagged:
                    # move to review dir preserving original subdir structure
                    sub_rel = os.path.relpath(dirpath, DATA_ROOT)
                    target_dir = os.path.join(REVIEW_DIR, sub_rel)
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, fname)
                    # if already exists, append a suffix
                    if os.path.exists(target_path):
                        base, ext = os.path.splitext(fname)
                        i = 1
                        while True:
                            candidate = os.path.join(target_dir, f"{base}_{i}{ext}")
                            if not os.path.exists(candidate):
                                target_path = candidate
                                break
                            i += 1
                    shutil.move(fpath, target_path)
                    flagged_count += 1

            except Exception as e:
                errors += 1
                rows.append([rel, os.path.basename(dirpath), -1, '', '', '', '', '', '', '', '', 'ERROR', traceback.format_exc()[:200]])

    # write CSV
    header = [
        'filepath_rel', 'class_folder', 'filesize_bytes', 'width', 'height',
        'top1', 'top1_prob', 'top2', 'top2_prob', 'top3', 'top3_prob',
        'flagged', 'flag_reasons'
    ]
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Finished. scanned={total}, flagged_moved={flagged_count}, errors={errors}")

    # clear TF session to free memory
    try:
        K.clear_session()
    except Exception:
        pass


if __name__ == '__main__':
    main()
