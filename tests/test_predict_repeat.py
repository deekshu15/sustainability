"""Small test script to POST the same image multiple times to /api/predict
Usage:
    python tests/test_predict_repeat.py --file static/assets/hero-traffic-s7FLP7fA.jpg --count 5

Requires `requests` (pip install requests)."""
import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("This test requires the 'requests' package. Install with: pip install requests")
    sys.exit(1)


def post_repeat(url, filepath, count=5, sleep=0.5):
    results = []
    for i in range(count):
        with open(filepath, 'rb') as fh:
            files = {'file': (filepath, fh, 'image/jpeg')}
            r = requests.post(url, files=files, timeout=30)
        try:
            obj = r.json()
        except Exception:
            obj = {'status_code': r.status_code, 'text': r.text}
        print(f"Run {i+1}: {json.dumps(obj)}")
        results.append(obj)
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:5000/api/predict')
    p.add_argument('--file', required=True)
    p.add_argument('--count', type=int, default=5)
    args = p.parse_args()
    post_repeat(args.url, args.file, args.count)
