
import csv, os, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
INPUT = os.path.join(DATA_ROOT, 'data_manifest_extended.csv')
OUTPUT = os.path.join(DATA_ROOT, 'data_manifest_reclass.csv')

DEFAULT_VEHICLE_KEYWORDS = [
    'car','truck','van','bus','ambulance','fire','engine','police','motorbike','motorcycle',
    'bicycle','taxi','minivan','passenger_car','go-kart','cart','wagon','jeep','cab','limousine',
    'tow_truck','snowplow','trailer','trailer_truck','recreational_vehicle','train','bullet_train',
    'tram','subway','airliner','plane','airplane','ship','speedboat','boat','ferry','yacht'
]

parser = argparse.ArgumentParser()
parser.add_argument('--whitelist', help='comma-separated additional whitelist keywords', default='')
args = parser.parse_args()

extra = [w.strip().lower() for w in args.whitelist.split(',') if w.strip()]
vehicle_keywords = set(DEFAULT_VEHICLE_KEYWORDS) | set(extra)

if not os.path.exists(INPUT):
    print('Input manifest not found:', INPUT)
    raise SystemExit(1)

rows_out = []
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames[:] if reader.fieldnames else []
    if 'nontraffic_flag' not in fieldnames:
        fieldnames += ['nontraffic_flag','nontraffic_reason']
    for r in reader:
        top1 = (r.get('top1') or '').lower()
        top2 = (r.get('top2') or '').lower()
        top3 = (r.get('top3') or '').lower()
        is_vehicle = False
        matched = []
        for lab in (top1, top2, top3):
            for kw in vehicle_keywords:
                if kw in lab:
                    is_vehicle = True
                    matched.append(kw)
                    break
            if is_vehicle:
                break
        r['nontraffic_flag'] = 'NO' if is_vehicle else 'YES'
        r['nontraffic_reason'] = ','.join(matched) if matched else 'no_vehicle_keyword_in_top3'
        rows_out.append(r)

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

count_non = sum(1 for r in rows_out if r['nontraffic_flag']=='YES')
count_yes = sum(1 for r in rows_out if r['nontraffic_flag']=='NO')
print('Wrote:', OUTPUT)
print('Counts -> nontraffic:', count_non, 'vehicle-like:', count_yes)
print('Used vehicle keywords:', sorted(list(vehicle_keywords))[:50], '...')
