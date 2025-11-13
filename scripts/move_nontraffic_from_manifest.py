
import os, csv, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
CSV_PATH = os.path.join(DATA_ROOT, 'data_manifest.csv')
OUT_CSV = os.path.join(DATA_ROOT, 'data_manifest_extended.csv')
REVIEW_DIR = os.path.join(DATA_ROOT, 'review', 'nontraffic')

VEHICLE_KEYWORDS = {
    'car','truck','van','bus','ambulance','fire','engine','police','motorbike','motorcycle',
    'bicycle','taxi','minivan','passenger_car','go-kart','cart','wagon','jeep','cab','limousine',
    'tow_truck','snowplow','trailer','trailer_truck','recreational_vehicle','train','bullet_train',
    'tram','subway','airliner','plane','airplane','ship','speedboat','boat','ferry','yacht','cart'
}

os.makedirs(REVIEW_DIR, exist_ok=True)

rows_out = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        filepath_rel = r['filepath_rel']
        top1 = (r.get('top1') or '').lower()
        top2 = (r.get('top2') or '').lower()
        top3 = (r.get('top3') or '').lower()
        # check if any of the top labels contains a vehicle keyword
        is_vehicle = False
        matched = []
        for lab in (top1, top2, top3):
            for kw in VEHICLE_KEYWORDS:
                if kw in lab:
                    is_vehicle = True
                    matched.append(kw)
                    break
            if is_vehicle:
                break
        nontraffic = 'NO' if is_vehicle else 'YES'
        reason = ','.join(matched) if matched else 'no_vehicle_keyword_in_top3'
        r['nontraffic_flag'] = nontraffic
        r['nontraffic_reason'] = reason
        rows_out.append(r)
        if nontraffic == 'YES':
            # move file to review folder preserving subdir
            src = os.path.join(DATA_ROOT, filepath_rel)
            src_dir = os.path.dirname(src)
            sub_rel = os.path.relpath(src_dir, DATA_ROOT)
            target_dir = os.path.join(REVIEW_DIR, sub_rel)
            os.makedirs(target_dir, exist_ok=True)
            if os.path.exists(src):
                target_path = os.path.join(target_dir, os.path.basename(src))
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(os.path.basename(src))
                    i = 1
                    while True:
                        cand = os.path.join(target_dir, f"{base}_{i}{ext}")
                        if not os.path.exists(cand):
                            target_path = cand
                            break
                        i += 1
                shutil.move(src, target_path)

# write extended csv
fieldnames = list(rows_out[0].keys()) if rows_out else []
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print('done. total_rows=', len(rows_out))
