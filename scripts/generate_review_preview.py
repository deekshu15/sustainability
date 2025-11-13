
import os
from PIL import Image
from pathlib import Path
import html

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
REVIEW_ROOT = DATA_ROOT / 'review'
THUMBS_ROOT = REVIEW_ROOT / '.thumbs'
GALLERY_HTML = DATA_ROOT / 'review_gallery.html'
MANUAL_HTML = DATA_ROOT / 'review_manual.html'
THUMB_SIZE = (200, 200)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def collect_images():
    items = []
    if not REVIEW_ROOT.exists():
        return items
    for dirpath, dirs, files in os.walk(REVIEW_ROOT):
        dirp = Path(dirpath)
        # skip thumbs folder if re-run
        if THUMBS_ROOT in dirp.parents or dirp == THUMBS_ROOT:
            continue
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                full = dirp / f
                rel = full.relative_to(DATA_ROOT)
                items.append((full, rel))
    items.sort(key=lambda x: str(x[1]))
    return items


def make_thumbnail(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert('RGB')
            im.thumbnail(THUMB_SIZE)
            im.save(dest, format='JPEG', quality=75)
    except Exception as e:
        # create a small placeholder
        with Image.new('RGB', THUMB_SIZE, (200,200,200)) as im:
            im.save(dest, format='JPEG')


def build_html(items):
    # gallery
    lines = [
        '<!doctype html>',
        '<html><head><meta charset="utf-8"><title>Review Gallery</title>',
        '<style>body{font-family:Arial,Helvetica,sans-serif} .grid{display:flex;flex-wrap:wrap} .item{margin:6px;text-align:center;width:220px}.item img{display:block;border:1px solid #ccc}</style>',
        '</head><body>',
        '<h1>Review Gallery — images under data/review</h1>',
        '<p>Thumbnails generated to <code>data/review/.thumbs/</code>.</p>',
        '<div class="grid">'
    ]
    for full, rel in items:
        thumb = THUMBS_ROOT / rel
        thumb_rel = thumb.relative_to(DATA_ROOT).as_posix()
        img_rel = rel.as_posix()
        lines.append('<div class="item">')
        lines.append(f'<a href="{html.escape(img_rel)}" target="_blank"><img src="{html.escape(thumb_rel)}" width="200"></a>')
        lines.append(f'<div style="font-size:12px">{html.escape(img_rel)}</div>')
        lines.append('</div>')
    lines.append('</div>')
    lines.append('</body></html>')
    GALLERY_HTML.write_text('\n'.join(lines), encoding='utf-8')

    # manual review HTML
    manual_lines = [
        '<!doctype html>',
        '<html><head><meta charset="utf-8"><title>Manual Review</title>',
        '<style>body{font-family:Arial,Helvetica,sans-serif} .grid{display:flex;flex-wrap:wrap} .item{margin:6px;text-align:center;width:220px}.item img{display:block;border:1px solid #ccc}</style>',
        '</head><body>',
        '<h1>Manual Review — mark images to archive/delete</h1>',
        '<p>Select images you want to archive (move to data/archive) and click Export decisions to download a JSON file that the provided script `scripts/apply_review_decisions.py` can process.</p>',
        '<button id="selectAll">Select all</button> <button id="clearAll">Clear</button> <button id="export">Export decisions (download JSON)</button>',
        '<div class="grid" id="grid">'
    ]
    for full, rel in items:
        thumb = THUMBS_ROOT / rel
        thumb_rel = thumb.relative_to(DATA_ROOT).as_posix()
        img_rel = rel.as_posix()
        manual_lines.append('<div class="item">')
        manual_lines.append(f'<img src="{html.escape(thumb_rel)}" width="200">')
        manual_lines.append(f'<div style="font-size:12px">{html.escape(img_rel)}</div>')
        manual_lines.append(f'<div><label><input type="checkbox" class="mark" data-path="{html.escape(img_rel)}"> Archive</label></div>')
        manual_lines.append('</div>')
    manual_lines.append('</div>')
    manual_lines.append('<script>')
    manual_lines.append('document.getElementById("selectAll").addEventListener("click", function(){document.querySelectorAll(".mark").forEach(c=>c.checked=true)});')
    manual_lines.append('document.getElementById("clearAll").addEventListener("click", function(){document.querySelectorAll(".mark").forEach(c=>c.checked=false)});')
    manual_lines.append('document.getElementById("export").addEventListener("click", function(){const arr=[];document.querySelectorAll(".mark").forEach(c=>{if(c.checked){arr.push({filepath_rel:c.dataset.path,action:"delete"})}});const blob=new Blob([JSON.stringify(arr,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="review_decisions.json";a.click();URL.revokeObjectURL(url);});')
    manual_lines.append('</script>')
    manual_lines.append('</body></html>')
    MANUAL_HTML.write_text('\n'.join(manual_lines), encoding='utf-8')


if __name__ == '__main__':
    items = collect_images()
    if not items:
        print('No images found under', REVIEW_ROOT)
        raise SystemExit(0)
    # generate thumbnails
    for full, rel in items:
        thumb = THUMBS_ROOT / rel
        if not thumb.exists():
            try:
                make_thumbnail(full, thumb)
            except Exception as e:
                print('thumb error', full, e)
    build_html(items)
    print('Wrote:', GALLERY_HTML)
    print('Wrote:', MANUAL_HTML)
