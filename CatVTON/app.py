# import gc
# import os
# import sys
# import base64
# import io
# import uuid
# import numpy as np
# import torch
# from PIL import Image, ImageFilter
# from flask import Flask, request, jsonify, render_template_string, send_from_directory
# from flask_cors import CORS

# # Ensure local imports work
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from model.cloth_masker import AutoMasker
# from model.pipeline import CatVTONPipeline

# app = Flask(__name__)
# CORS(app)


# from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# app = Flask(__name__)

# # Configuration
# SERVER_IP = "192.168.50.211"
# PORT = 5000
# RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # Global Models
# PIPELINE = None
# MASKER   = None
# DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# HTML = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8"/>
#     <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
#     <title>Virtual Try-On</title>
#     <style>
#         * { box-sizing: border-box; margin: 0; padding: 0; }
#         body { font-family: 'Segoe UI', sans-serif; background: #f0ede8; min-height: 100vh; padding: 32px 16px; color: #1a1a18; }
#         h1 { text-align: center; font-size: 2rem; font-weight: 300; letter-spacing: 0.08em; margin-bottom: 6px; }
#         .subtitle { text-align: center; font-size: 12px; color: #999; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 36px; }
#         .container { max-width: 1100px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
#         @media (max-width: 700px) { .container { grid-template-columns: 1fr; } }
#         .card { background: #fff; border: 1px solid #e0dbd2; padding: 24px; }
#         .card h2 { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.14em; color: #999; margin-bottom: 16px; }
#         .upload-box { border: 1.5px dashed #ccc; background: #faf9f6; width: 100%; height: 240px; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; position: relative; overflow: hidden; }
#         .upload-box:hover { border-color: #888; }
#         .upload-box input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
#         .upload-box img { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
#         .upload-box .placeholder { text-align: center; color: #bbb; font-size: 13px; pointer-events: none; }
#         .upload-box .placeholder span { font-size: 32px; display: block; margin-bottom: 8px; }
#         .options { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }
#         .option-group { flex: 1; min-width: 140px; }
#         .option-group label { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: #888; display: block; margin-bottom: 6px; }
#         select { width: 100%; padding: 8px 10px; border: 1px solid #ddd; background: #faf9f6; font-size: 13px; color: #1a1a18; }
#         #run-btn { width: 100%; padding: 14px; background: #1a1a18; color: #f0ede8; border: none; font-size: 12px; font-weight: 500; letter-spacing: 0.16em; text-transform: uppercase; cursor: pointer; margin-top: 20px; transition: background 0.2s; }
#         #run-btn:hover { background: #444; }
#         #run-btn:disabled { background: #aaa; cursor: not-allowed; }
#         .result-panel { background: #fff; border: 1px solid #e0dbd2; padding: 24px; display: flex; flex-direction: column; }
#         .result-panel h2 { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.14em; color: #999; margin-bottom: 16px; }
#         #result-box { flex: 1; background: #f5f3ef; border: 1px solid #e8e4dc; display: flex; align-items: center; justify-content: center; min-height: 400px; }
#         #result-img { max-width: 100%; max-height: 580px; object-fit: contain; display: none; }
#         #placeholder-text { text-align: center; color: #ccc; font-size: 13px; }
#         #placeholder-text span { font-size: 40px; display: block; margin-bottom: 8px; }
#         .spinner { display: none; flex-direction: column; align-items: center; gap: 12px; color: #888; font-size: 13px; }
#         .spinner.active { display: flex; }
#         .spin { width: 36px; height: 36px; border: 3px solid #ddd; border-top-color: #1a1a18; border-radius: 50%; animation: spin 0.8s linear infinite; }
#         @keyframes spin { to { transform: rotate(360deg); } }
#         #download-btn { display: none; margin-top: 14px; width: 100%; padding: 11px; background: transparent; border: 1px solid #1a1a18; color: #1a1a18; font-size: 11px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; cursor: pointer; text-align: center; text-decoration: none; }
#         #download-btn:hover { background: #f0ede8; }
#         #status-msg { font-size: 12px; color: #e05c5c; margin-top: 10px; text-align: center; min-height: 18px; }
#         #success-msg { font-size: 12px; color: #2e7d32; margin-top: 10px; text-align: center; min-height: 18px; }
#     </style>
# </head>
# <body>
#     <h1>Virtual Try&#8209;On</h1>
#     <p class="subtitle">Upload &nbsp;·&nbsp; Select &nbsp;·&nbsp; Generate</p>
#     <div class="container">
#         <div class="card">
#             <h2>01 — Inputs</h2>
#             <label style="font-size:12px;color:#888;display:block;margin-bottom:6px;">Person photo</label>
#             <div class="upload-box">
#                 <input type="file" accept="image/*" id="person-input" onchange="previewImage(this,'person-preview','person-placeholder')"/>
#                 <img id="person-preview"/>
#                 <div class="placeholder" id="person-placeholder"><span>🧍</span>Click to upload person photo</div>
#             </div>
#             <label style="font-size:12px;color:#888;display:block;margin:16px 0 6px;">Garment image</label>
#             <div class="upload-box">
#                 <input type="file" accept="image/*" id="cloth-input" onchange="previewImage(this,'cloth-preview','cloth-placeholder')"/>
#                 <img id="cloth-preview"/>
#                 <div class="placeholder" id="cloth-placeholder"><span>👕</span>Click to upload garment</div>
#             </div>
#             <div class="options">
#                 <div class="option-group">
#                     <label>Garment type</label>
#                     <select id="cloth-type">
#                         <option value="upper">Upper body</option>
#                         <option value="lower">Lower body</option>
#                         <option value="overall">Full outfit</option>
#                     </select>
#                 </div>
#                 <div class="option-group">
#                     <label>Background</label>
#                     <select id="bg-option">
#                         <option value="original">Original</option>
#                         <option value="white">White</option>
#                         <option value="black">Black</option>
#                         <option value="blur">Blur</option>
#                         <option value="model">Model default</option>
#                     </select>
#                 </div>
#             </div>
#             <button id="run-btn" onclick="runTryon()">Generate Try-On</button>
#             <p id="status-msg"></p>
#         </div>

#         <div class="result-panel">
#             <h2>02 — Result</h2>
#             <div id="result-box">
#                 <div class="spinner" id="spinner">
#                     <div class="spin"></div>
#                     <span id="spinner-text">Processing...</span>
#                 </div>
#                 <img id="result-img"/>
#                 <div id="placeholder-text"><span>✨</span>Result will appear here</div>
#             </div>
#             <p id="success-msg"></p>
#             <a id="download-btn" target="_blank">Download Result</a>
#         </div>
#     </div>

#     <script>
#         function previewImage(input, previewId, placeholderId) {
#             const file = input.files[0];
#             if (!file) return;
#             const reader = new FileReader();
#             reader.onload = e => {
#                 const p = document.getElementById(previewId);
#                 p.src = e.target.result;
#                 p.style.display = 'block';
#                 document.getElementById(placeholderId).style.display = 'none';
#             };
#             reader.readAsDataURL(file);
#         }

#         function fileToBase64(file) {
#             return new Promise((res, rej) => {
#                 const r = new FileReader();
#                 r.onload = () => res(r.result.split(',')[1]);
#                 r.onerror = rej;
#                 r.readAsDataURL(file);
#             });
#         }

#         const msgs = ['Preprocessing...','Generating mask...','Running diffusion (~1-2 min)...','Finalizing...'];
#         let idx = 0, timer = null;

#         function startSpinner() {
#             document.getElementById('placeholder-text').style.display = 'none';
#             document.getElementById('result-img').style.display = 'none';
#             document.getElementById('download-btn').style.display = 'none';
#             document.getElementById('success-msg').textContent = '';
#             document.getElementById('status-msg').textContent = '';
#             idx = 0;
#             document.getElementById('spinner-text').textContent = msgs[0];
#             document.getElementById('spinner').classList.add('active');
#             timer = setInterval(() => {
#                 idx = (idx + 1) % msgs.length;
#                 document.getElementById('spinner-text').textContent = msgs[idx];
#             }, 8000);
#         }

#         function stopSpinner() {
#             clearInterval(timer);
#             document.getElementById('spinner').classList.remove('active');
#         }

#         async function runTryon() {
#             const personFile = document.getElementById('person-input').files[0];
#             const clothFile  = document.getElementById('cloth-input').files[0];
#             if (!personFile || !clothFile) { 
#                 document.getElementById('status-msg').textContent = 'Please upload both photos.'; 
#                 return; 
#             }

#             document.getElementById('run-btn').disabled = true;
#             startSpinner();

#             try {
#                 const [p64, c64] = await Promise.all([fileToBase64(personFile), fileToBase64(clothFile)]);

#                 const resp = await fetch('/tryon', {
#                     method: 'POST',
#                     headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify({
#                         person: p64, cloth: c64,
#                         cloth_type: document.getElementById('cloth-type').value,
#                         bg_option:  document.getElementById('bg-option').value,
#                     }),
#                 });

#                 const data = await resp.json();
#                 stopSpinner();

#                 if (data.error) {
#                     document.getElementById('status-msg').textContent = 'Error: ' + data.error;
#                     document.getElementById('placeholder-text').style.display = 'block';
#                 } else {
#                     const img = document.getElementById('result-img');
#                     // Add cache buster to ensure the image refreshes
#                     img.src = data.url + '?t=' + new Date().getTime();
#                     img.style.display = 'block';

#                     const dl = document.getElementById('download-btn');
#                     dl.href = data.url;
#                     dl.style.display = 'block';

#                     document.getElementById('success-msg').textContent = '✓ Generation Successful';
#                 }
#             } catch(err) {
#                 stopSpinner();
#                 document.getElementById('status-msg').textContent = 'Failed to connect to server.';
#                 document.getElementById('placeholder-text').style.display = 'block';
#             }
#             document.getElementById('run-btn').disabled = false;
#         }
#     </script>
# </body>
# </html>
# """

# @app.route("/")
# def index():
#     return render_template_string(HTML)

# @app.route("/results/<path:filename>")
# def serve_result(filename):
#     # as_attachment=False helps display in browser
#     return send_from_directory(RESULTS_DIR, filename, as_attachment=False)

# @app.route("/tryon", methods=["POST"])
# def tryon():
#     try:
#         data       = request.get_json()
#         person     = Image.open(io.BytesIO(base64.b64decode(data["person"]))).convert("RGB")
#         cloth      = Image.open(io.BytesIO(base64.b64decode(data["cloth"]))).convert("RGB")
#         cloth_type = data.get("cloth_type", "upper")
#         bg_option  = data.get("bg_option",  "original")

#         # 1. Preprocessing
#         W, H     = 768, 1024
#         person_r = resize_and_crop(person, (W, H))
#         cloth_r  = resize_and_padding(cloth, (W, H))
        
#         # 2. Masking
#         mask_result = MASKER(person_r, cloth_type)
#         mask = mask_result["mask"]

#         # 3. Inference
#         gen = torch.Generator(device=DEVICE).manual_seed(42)
#         out = PIPELINE(
#             image=person_r, condition_image=cloth_r, mask=mask,
#             num_inference_steps=50, guidance_scale=2.5, generator=gen,
#         )
        
#         # Handle different return types from pipeline
#         if hasattr(out, "images"):
#             result = out.images[0]
#         elif isinstance(out, (list, tuple)):
#             result = out[0]
#         else:
#             result = out

#         # 4. Post-processing (Background handling)
#         mask_np  = np.array(mask.convert("L"))
#         mask_bin = (mask_np > 128).astype(np.uint8)[:, :, None]
#         res_np   = np.array(result)
#         per_np   = np.array(person_r)

#         if bg_option == "original":
#             final_np = (per_np * (1 - mask_bin) + res_np * mask_bin).astype(np.uint8)
#             final = Image.fromarray(final_np)
#         elif bg_option == "white":
#             final_np = (res_np * mask_bin + np.full_like(res_np, 255) * (1 - mask_bin)).astype(np.uint8)
#             final = Image.fromarray(final_np)
#         elif bg_option == "black":
#             final = Image.fromarray((res_np * mask_bin).astype(np.uint8))
#         elif bg_option == "blur":
#             bl_img = person_r.filter(ImageFilter.GaussianBlur(radius=10))
#             bl_np = np.array(bl_img)
#             final_np = (res_np * mask_bin + bl_np * (1 - mask_bin)).astype(np.uint8)
#             final = Image.fromarray(final_np)
#         else:
#             final = result

#         # 5. Save & Return
#         filename  = f"result_{uuid.uuid4().hex[:8]}.png"
#         save_path = os.path.join(RESULTS_DIR, filename)
#         final.save(save_path)
#         print(f"Saved → {save_path}")

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         # RETURN FULL URL to avoid WireGuard relative path confusion
#         full_url = f"http://{SERVER_IP}:{PORT}/results/{filename}"
#         return jsonify({"url": full_url, "filename": filename})

#     except Exception as e:
#         import traceback; traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# def find_cached_model():
#     cache = os.path.expanduser("~/.cache/huggingface/hub/models--zhengchong--CatVTON/snapshots")
#     if os.path.exists(cache):
#         snaps = sorted(os.listdir(cache))
#         if snaps:
#             return os.path.join(cache, snaps[0])
#     return "zhengchong/CatVTON"

# def load_models():
#     global PIPELINE, MASKER
#     resume = find_cached_model()
#     print(f"Device : {DEVICE}\nModel  : {resume}\n")
#     dtype = init_weight_dtype("fp16")
    
#     print("Loading pipeline...")
#     PIPELINE = CatVTONPipeline(
#         base_ckpt="booksforcharlie/stable-diffusion-inpainting",
#         attn_ckpt=resume, attn_ckpt_version="mix",
#         weight_dtype=dtype, use_tf32=True, device=DEVICE,
#     )
    
#     print("Loading AutoMasker...")
#     MASKER = AutoMasker(
#         densepose_ckpt=os.path.join(resume, "DensePose"),
#         schp_ckpt=os.path.join(resume, "SCHP"),
#         device=DEVICE,
#     )
#     print("Ready!\n")

# if __name__ == "__main__":
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#     load_models()
#     print("=" * 40)
#     print(f"  Access the app at: http://{SERVER_IP}:{PORT}")
#     print("=" * 40 + "\n")
#     app.run(host="0.0.0.0", port=PORT, debug=False)


import gc
import os
import sys
import base64
import io
import uuid
import numpy as np
import torch
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# ✅ CREATE APP ONLY ONCE
app = Flask(__name__)
CORS(app)   # ✅ enable React connection

# Configuration
SERVER_IP = "192.168.50.211"
PORT = 5000
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global Models
PIPELINE = None
MASKER   = None
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- ROUTES ---------------- #

@app.route("/")
def index():
    return "Backend is running"

@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=False)

@app.route("/tryon", methods=["POST"])
def tryon():
    try:
        data = request.get_json()

        person = Image.open(io.BytesIO(base64.b64decode(data["person"]))).convert("RGB")
        cloth  = Image.open(io.BytesIO(base64.b64decode(data["cloth"]))).convert("RGB")

        cloth_type = data.get("cloth_type", "upper")
        bg_option  = data.get("bg_option", "original")

        # 1. Resize
        W, H = 768, 1024
        person_r = resize_and_crop(person, (W, H))
        cloth_r  = resize_and_padding(cloth, (W, H))

        # 2. Mask
        mask = MASKER(person_r, cloth_type)["mask"]

        # 3. Model inference
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        out = PIPELINE(
            image=person_r,
            condition_image=cloth_r,
            mask=mask,
            num_inference_steps=50,
            guidance_scale=2.5,
            generator=gen,
        )

        if hasattr(out, "images"):
            result = out.images[0]
        elif isinstance(out, (list, tuple)):
            result = out[0]
        else:
            result = out

        # 4. Background processing
        mask_np  = np.array(mask.convert("L"))
        mask_bin = (mask_np > 128).astype(np.uint8)[:, :, None]

        res_np = np.array(result)
        per_np = np.array(person_r)

        if bg_option == "original":
            final_np = (per_np * (1 - mask_bin) + res_np * mask_bin).astype(np.uint8)
        elif bg_option == "white":
            final_np = (res_np * mask_bin + np.full_like(res_np, 255) * (1 - mask_bin)).astype(np.uint8)
        elif bg_option == "black":
            final_np = (res_np * mask_bin).astype(np.uint8)
        elif bg_option == "blur":
            blur_img = person_r.filter(ImageFilter.GaussianBlur(radius=10))
            blur_np = np.array(blur_img)
            final_np = (res_np * mask_bin + blur_np * (1 - mask_bin)).astype(np.uint8)
        else:
            final_np = res_np

        final = Image.fromarray(final_np)

        # 5. Save result
        filename = f"result_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(RESULTS_DIR, filename)
        final.save(path)

        print(f"Saved → {path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # return full URL
        url = f"http://{SERVER_IP}:{PORT}/results/{filename}"
        return jsonify({"url": url})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- MODEL LOADING ---------------- #

def find_cached_model():
    cache = os.path.expanduser("~/.cache/huggingface/hub/models--zhengchong--CatVTON/snapshots")
    if os.path.exists(cache):
        snaps = sorted(os.listdir(cache))
        if snaps:
            return os.path.join(cache, snaps[0])
    return "zhengchong/CatVTON"

def load_models():
    global PIPELINE, MASKER

    resume = find_cached_model()
    print(f"Device: {DEVICE}\nModel: {resume}\n")

    dtype = init_weight_dtype("fp16")

    print("Loading pipeline...")
    PIPELINE = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=resume,
        attn_ckpt_version="mix",
        weight_dtype=dtype,
        use_tf32=True,
        device=DEVICE,
    )

    print("Loading AutoMasker...")
    MASKER = AutoMasker(
        densepose_ckpt=os.path.join(resume, "DensePose"),
        schp_ckpt=os.path.join(resume, "SCHP"),
        device=DEVICE,
    )

    print("✅ Models Ready!\n")

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    load_models()

    print("=" * 40)
    print(f"Server running at: http://{SERVER_IP}:{PORT}")
    print("=" * 40)

    app.run(host="0.0.0.0", port=PORT, debug=False)