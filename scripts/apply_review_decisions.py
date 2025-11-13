
import sys, os, json, shutil
from datetime import datetime

if len(sys.argv) < 2:
    print('Usage: python scripts/apply_review_decisions.py review_decisions.json')
    raise SystemExit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
ARCHIVE_ROOT = os.path.join(DATA_ROOT, 'archive')

decisions_path = sys.argv[1]
if not os.path.exists(decisions_path):
    print('decisions file not found:', decisions_path)
    raise SystemExit(1)

with open(decisions_path, 'r', encoding='utf-8') as f:
    decisions = json.load(f)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
archive_dir = os.path.join(ARCHIVE_ROOT, timestamp)

moved = 0
errors = 0
for item in decisions:
    rel = item.get('filepath_rel')
    action = item.get('action')
    if not rel:
        continue
    src = os.path.join(DATA_ROOT, rel)
    if not os.path.exists(src):
        print('not found:', src)
        errors += 1
        continue
    if action == 'delete' or action == 'archive':
        target = os.path.join(archive_dir, os.path.dirname(rel))
        os.makedirs(target, exist_ok=True)
        tgt_path = os.path.join(target, os.path.basename(rel))
        # avoid overwrite
        if os.path.exists(tgt_path):
            base, ext = os.path.splitext(os.path.basename(rel))
            i = 1
            while True:
                cand = os.path.join(target, f"{base}_{i}{ext}")
                if not os.path.exists(cand):
                    tgt_path = cand
                    break
                i += 1
        shutil.move(src, tgt_path)
        moved += 1
        print('moved', src, '->', tgt_path)
    else:
        print('skipping (unknown action):', rel, action)

print(f'done. moved={moved}, errors={errors}. archive dir: {archive_dir}')
