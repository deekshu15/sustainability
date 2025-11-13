import os, zipfile
from pathlib import Path
DATA = Path('data')
THUMBS = DATA / 'review' / '.thumbs'
OUT = DATA / 'review_samples.zip'
files = []
for dirpath, dirs, fs in os.walk(THUMBS):
    for f in fs:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            files.append(os.path.join(dirpath,f))
files.sort()
sel = files[:10]
if not sel:
    print('No thumbnails found under', THUMBS)
else:
    with zipfile.ZipFile(OUT, 'w') as z:
        for p in sel:
            arcname = os.path.relpath(p, DATA)
            z.write(p, arcname)
    print('Wrote', OUT, 'with', len(sel), 'files')
    for p in sel:
        print(' ', os.path.relpath(p, DATA))
