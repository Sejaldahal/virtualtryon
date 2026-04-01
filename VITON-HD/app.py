"""
Flask backend for VITON-HD Virtual Try-On Web Interface
No uploads — selects from images already in your dataset.
Run: python app.py
"""

import os
import sys
import random
import subprocess
import traceback
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR      = r'E:\Coding\Python\VITON\AI Virtual Tryon\VITON-HD\datasets\test'
CHECKPOINT_DIR   = os.path.join(PROJECT_DIR, 'checkpoints')
ALIAS_CHECKPOINT = os.path.join(PROJECT_DIR, 'checkpoints', 'alias_final.pth')

RESULTS_FOLDER = os.path.join(PROJECT_DIR, '_web_results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


@app.route('/')
def index():
    return send_file(os.path.join(PROJECT_DIR, 'index.html'))


@app.route('/list-images')
def list_images():
    """Return 10 random person and cloth images from the dataset."""
    def get_images(subfolder, n=10):
        folder = os.path.join(DATASET_DIR, subfolder)
        if not os.path.exists(folder):
            return []
        all_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        return random.sample(all_files, min(n, len(all_files)))

    return jsonify({
        'persons': get_images('image'),
        'cloths':  get_images('cloth'),
    })


@app.route('/preview/<folder>/<filename>')
def preview(folder, filename):
    """Serve a dataset image for browser preview."""
    if folder not in ('image', 'cloth'):
        return 'Not allowed', 403
    path = os.path.join(DATASET_DIR, folder, filename)
    if not os.path.exists(path):
        return 'Not found', 404
    return send_file(path, mimetype='image/jpeg')


@app.route('/tryon', methods=['POST'])
def tryon():
    """Run inference using filenames that already exist in the dataset."""
    data            = request.get_json()
    person_filename = (data or {}).get('person', '').strip()
    cloth_filename  = (data or {}).get('cloth',  '').strip()

    if not person_filename or not cloth_filename:
        return jsonify({'error': 'person and cloth filenames are required'}), 400

    # Verify files exist
    if not os.path.exists(os.path.join(DATASET_DIR, 'image', person_filename)):
        return jsonify({'error': f'Person not found: {person_filename}'}), 404
    if not os.path.exists(os.path.join(DATASET_DIR, 'cloth', cloth_filename)):
        return jsonify({'error': f'Cloth not found: {cloth_filename}'}), 404

    person_base     = os.path.splitext(person_filename)[0]
    cloth_base      = os.path.splitext(cloth_filename)[0]
    output_filename = f"{person_base}_{cloth_base}_finetuned.jpg"
    output_path     = os.path.join(RESULTS_FOLDER, output_filename)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, 'full_res_finetune_output.py'),
        '--person',           person_filename,
        '--cloth',            cloth_filename,
        '--output',           output_path,
        '--dataset_dir',      DATASET_DIR,
        '--checkpoint_dir',   CHECKPOINT_DIR,
        '--alias_checkpoint', ALIAS_CHECKPOINT,
    ]

    print(f"\n[VITON] {person_filename} + {cloth_filename}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,
            cwd=PROJECT_DIR,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        )

        print("[VITON] STDOUT:", result.stdout[-2000:])
        if result.stderr:
            print("[VITON] STDERR:", result.stderr[-2000:])

        if result.returncode != 0:
            return jsonify({
                'error': 'Generation failed',
                'details': (result.stderr or result.stdout or 'Unknown error')[-3000:]
            }), 500

        if not os.path.exists(output_path):
            candidates = [
                f for f in os.listdir(RESULTS_FOLDER)
                if person_base in f and cloth_base in f
            ]
            if candidates:
                output_path = os.path.join(RESULTS_FOLDER, sorted(candidates)[-1])
            else:
                return jsonify({'error': 'Output not found.', 'stdout': result.stdout[-2000:]}), 500

        return send_file(output_path, mimetype='image/jpeg')

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Timed out (>10 minutes)'}), 504
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  VITON-HD Try-On Server  ->  http://localhost:5000")
    print(f"  Dataset : {DATASET_DIR}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)