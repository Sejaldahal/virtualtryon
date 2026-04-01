import { useEffect, useRef, useState } from "react";

// ── Dataset ──────────────────────────────────────────────────────────────────
const DATASET = {
  models: [
    { id: "m1", label: "model1", url: new URL("../components/dataset/models/00674_00.jpg", import.meta.url).href },
    { id: "m2", label: "model2", url: new URL("../components/dataset/models/00124_00.jpg", import.meta.url).href },
    { id: "m3", label: "model3", url: new URL("../components/dataset/models/00153_00.jpg", import.meta.url).href },
    { id: "m4", label: "model4", url: new URL("../components/dataset/models/00586_00.jpg", import.meta.url).href },
  ],
  clothes: [
    { id: "c1", label: "1", url: new URL("../components/dataset/clothes/00000_00.jpg", import.meta.url).href },
    { id: "c2", label: "2", url: new URL("../components/dataset/clothes/00024_00.jpg", import.meta.url).href },
    { id: "c3", label: "3", url: new URL("../components/dataset/clothes/traditional.jpg", import.meta.url).href },
    { id: "c4", label: "4", url: new URL("../components/dataset/clothes/00134_00.jpg", import.meta.url).href },
  ],
};

// ── Config ───────────────────────────────────────────────────────────────────
// ✏️ UPDATE THESE when your servers change
const TRYON_API = "http://192.168.50.211:5000";                       // College GPU — try-on
const CLOTH_API = "https://unpositive-nondeaf-sari.ngrok-free.dev";  // Colab ngrok — cloth gen

// ── Helpers ──────────────────────────────────────────────────────────────────
const resizeImage = (file: File, maxW = 512, maxH = 512): Promise<string> =>
  new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      let { width: w, height: h } = img;
      if (w > maxW) { h = (h * maxW) / w; w = maxW; }
      if (h > maxH) { w = (w * maxH) / h; h = maxH; }
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      c.getContext("2d")!.drawImage(img, 0, 0, w, h);
      resolve(c.toDataURL("image/jpeg"));
    };
    img.src = URL.createObjectURL(file);
  });

async function urlToBase64(url: string): Promise<string> {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve) => {
    const r = new FileReader();
    r.onloadend = () => {
      const result = r.result as string;
      const idx = result.indexOf(",");
      resolve(result.slice(idx + 1));
    };
    r.readAsDataURL(blob);
  });
}

// Safely extract raw base64 from either a data URI or plain base64 string
function extractBase64(src: string): string {
  if (src.startsWith("data:")) {
    const idx = src.indexOf(",");
    return src.slice(idx + 1);
  }
  return src;
}

// Normalise any image string into a displayable data URI
function toDataURI(src: string, mime = "image/png"): string {
  if (src.startsWith("data:")) return src;
  return `data:${mime};base64,${src}`;
}

// ── Types ────────────────────────────────────────────────────────────────────
type Asset = { id: string; label: string; url: string };

// ── Component ────────────────────────────────────────────────────────────────
export default function TryOn() {
  const [personAsset, setPersonAsset]         = useState<Asset | null>(null);
  const [clothAsset, setClothAsset]           = useState<Asset | null>(null);
  const [customPerson, setCustomPerson]       = useState<string | null>(null);
  const [customCloth, setCustomCloth]         = useState<string | null>(null);

  const [generatedCloth, setGeneratedCloth]   = useState<string | null>(null);
  const [prompt, setPrompt]                   = useState("");
  const [loading, setLoading]                 = useState(false);
  const [generatingCloth, setGeneratingCloth] = useState(false);
  const [approved, setApproved]               = useState(false);
  const [result, setResult]                   = useState<string | null>(null);
  const [step, setStep]                       = useState<1 | 2 | 3>(1);
  const [genError, setGenError]               = useState<string | null>(null);
  const [downloading, setDownloading]         = useState(false);
  const [isFullscreen, setIsFullscreen]       = useState(false);

  const personRef = useRef<HTMLInputElement>(null);
  const clothRef  = useRef<HTMLInputElement>(null);

  // ── Derived image sources ──
  const personSrc   = customPerson ?? personAsset?.url ?? null;
  const personLabel = customPerson ? "Custom" : (personAsset?.label ?? "—");

  const activeClothesSrc = customCloth ?? clothAsset?.url ?? null;
  const activeClothLabel = approved ? "AI Generated" : (customCloth ? "Custom" : (clothAsset?.label ?? "—"));

  // While pending approval show generated preview; after approval show approved cloth
  const garmentPreviewSrc = (generatedCloth && !approved) ? generatedCloth : activeClothesSrc;

  const canTryOn = !!personSrc && !!activeClothesSrc && !loading && !generatingCloth;

  // ── Select preset model ──
  const selectModel = (m: Asset) => {
    if (personAsset?.id === m.id) {
      // Toggle: deselect if already selected
      setPersonAsset(null);
      setCustomPerson(null);
    } else {
      // Select new model
      setPersonAsset(m);
      setCustomPerson(null);
      setStep(s => s < 2 ? 2 : s);
    }
  };

  // ── Select preset cloth ──
  const selectCloth = (c: Asset) => {
    if (clothAsset?.id === c.id) {
      // Toggle: deselect if already selected
      setClothAsset(null);
      setCustomCloth(null);
      setGeneratedCloth(null);
      setApproved(false);
      setGenError(null);
    } else {
      // Select new cloth
      setClothAsset(c);
      setCustomCloth(null);
      setGeneratedCloth(null);
      setApproved(false);
      setGenError(null);
      setStep(s => s < 3 ? 3 : s);
    }
  };

  // ── Upload handler ──
  const handleUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
    type: "person" | "cloth"
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const resized = await resizeImage(file);
    if (type === "person") {
      setCustomPerson(resized);
      setPersonAsset(null);
      setStep(s => s < 2 ? 2 : s);
    } else {
      setCustomCloth(resized);
      setClothAsset(null);
      setGeneratedCloth(null);
      setApproved(false);
      setGenError(null);
    }
    // Reset file input so same file can be re-uploaded
    e.target.value = "";
  };

  // ── Generate clothing via Colab + ngrok ──
  const generateClothing = async () => {
    if (!prompt.trim()) return alert("Write a clothing description first.");
    setGeneratingCloth(true);
    setGeneratedCloth(null);
    setApproved(false);
    setGenError(null);
    // Clear any previously approved generated cloth
    if (approved) {
      setCustomCloth(null);
    }

    try {
      const res = await fetch(`${CLOTH_API}/generate_cloth`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",  // prevents ngrok interstitial page
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          num_steps: 30,
          guidance_scale: 7.5,
          seed: Math.floor(Math.random() * 99999),
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error ${res.status}: ${text.slice(0, 200)}`);
      }

      const data = await res.json();

      if (data.error) {
        setGenError(data.error);
        console.error("Generation error from server:", data.error);
      } else if (data.image) {
        // Handles both full data URI and raw base64
        const src = toDataURI(data.image, "image/png");
        setGeneratedCloth(src);
      } else {
        setGenError("Server returned no image. Check Colab logs.");
      }

    } catch (err: any) {
      console.error("Cloth generation fetch error:", err);
      setGenError(err.message ?? "Could not reach generation server.");
    }

    setGeneratingCloth(false);
  };

  // ── Approve generated cloth for try-on ──
  const approveCloth = () => {
    if (!generatedCloth) return;
    setCustomCloth(generatedCloth);  // store as active cloth
    setClothAsset(null);
    setApproved(true);
    setGenError(null);
    setStep(s => s < 3 ? 3 : s);
  };

  // ── Redo generation ──
  const redoGenerate = () => {
    setGeneratedCloth(null);
    setApproved(false);
    setGenError(null);
    if (approved) setCustomCloth(null);
    generateClothing();
  };

  // ── Download result image reliably (works better than plain anchor for API URLs) ──
  const downloadResult = async () => {
    if (!result || downloading) return;
    setDownloading(true);

    try {
      const res = await fetch(result);
      if (!res.ok) {
        throw new Error(`Download failed with status ${res.status}`);
      }

      const blob = await res.blob();
      const fileExt = blob.type.includes("png") ? "png" : "jpg";
      const objectUrl = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `tryon-result.${fileExt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();

      URL.revokeObjectURL(objectUrl);
    } catch (err: any) {
      console.error("Download failed:", err);
      alert("Could not download this image directly. This can happen when the image server blocks cross-origin downloads.");
    } finally {
      setDownloading(false);
    }
  };

  useEffect(() => {
    if (!isFullscreen) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsFullscreen(false);
    };

    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isFullscreen]);

  // ── Try-on via college GPU Flask ──
  const generatePreview = async () => {
    if (!personSrc || !activeClothesSrc) return alert("Select a model and a garment first.");
    setLoading(true);
    setResult(null);

    try {
      const personBase64 = personSrc.startsWith("data:")
        ? extractBase64(personSrc)
        : await urlToBase64(personSrc);

      const clothBase64 = activeClothesSrc.startsWith("data:")
        ? extractBase64(activeClothesSrc)
        : await urlToBase64(activeClothesSrc);

      const res = await fetch(`${TRYON_API}/tryon`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person: personBase64,
          cloth: clothBase64,
          cloth_type: "upper",
          bg_option: "original",
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Try-on server error ${res.status}: ${text.slice(0, 200)}`);
      }

      const data = await res.json();
      if (data.error) {
        alert(`Try-on failed: ${data.error}`);
      } else if (data.url) {
        setResult(data.url + "?t=" + Date.now());
      } else if (data.image) {
        setResult(toDataURI(data.image, "image/jpeg"));
      } else {
        alert("No result returned from try-on server.");
      }

    } catch (err: any) {
      console.error("Try-on error:", err);
      alert(`Failed to connect to try-on backend.\n\n${err.message}`);
    }

    setLoading(false);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", background: "#26262a", minHeight: "100vh", color: "#f0ede8" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,300&family=DM+Serif+Display:ital@0;1&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        .card-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        @media(max-width:640px){ .card-grid { grid-template-columns: repeat(2,1fr); } }
        .thumb {
          border-radius: 12px; overflow: hidden; cursor: pointer;
          border: 2px solid transparent; transition: border-color .2s, transform .2s, box-shadow .2s;
          position: relative; aspect-ratio: 3/4; background: #1a1a1f;
        }
        .thumb:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,.4); }
        .thumb.selected { border-color: #bc9968; box-shadow: 0 0 0 4px rgba(200,169,126,.15); }
        .thumb img { width:100%; height:100%; object-fit:cover; display:block; }
        .thumb .badge {
          position:absolute; bottom:8px; left:8px; right:8px;
          background:rgba(13,13,15,.75); backdrop-filter:blur(6px);
          border-radius:6px; padding:4px 8px;
          font-size:11px; font-weight:500; letter-spacing:.04em; text-align:center;
          color:#f0ede8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        }
        .thumb .check {
          position:absolute; top:8px; right:8px; width:22px; height:22px; border-radius:50%;
          background:#c8a97e; display:flex; align-items:center; justify-content:center;
          font-size:11px; opacity:0; transition:opacity .2s;
        }
        .thumb.selected .check { opacity:1; }
        .upload-tile {
          border-radius:12px; border:2px dashed #2e2e35; cursor:pointer;
          display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px;
          aspect-ratio:3/4; background:#13131a; transition: border-color .2s, background .2s;
          font-size:12px; color:#6b6b7a; font-weight:500;
        }
        .upload-tile:hover { border-color:#c8a97e; background:#18181f; color:#c8a97e; }
        .section-title { font-family:'DM Serif Display', serif; font-size:18px; letter-spacing:-.01em; color:#f0ede8; margin-bottom:14px; }
        .section-tag {
          display:inline-flex; align-items:center; gap:6px;
          background:#1e1e26; border:1px solid #2a2a35; border-radius:20px;
          padding:4px 12px; font-size:11px; font-weight:500; letter-spacing:.08em;
          color:#8b8b9e; text-transform:uppercase; margin-bottom:10px;
        }
        .dot { width:6px;height:6px;border-radius:50%;background:#c8a97e; }
        .result-panel {
          border-radius:16px; border:1px solid #1e1e26; background:#101015; overflow:hidden;
          display:flex; align-items:center; justify-content:center; min-height:420px; position:relative;
        }
        .result-img { max-height:420px; object-fit:contain; border-radius:12px; }
        .btn-primary {
          background: linear-gradient(135deg, #c8a97e, #a8865a); color: #0d0d0f;
          font-weight:700; font-size:14px; border:none; border-radius:12px;
          padding:14px 28px; cursor:pointer; width:100%; letter-spacing:.02em;
          transition: opacity .2s, transform .15s; font-family:'DM Sans',sans-serif;
        }
        .btn-primary:hover:not(:disabled) { opacity:.9; transform:translateY(-1px); }
        .btn-primary:disabled { opacity:.35; cursor:not-allowed; }
        .btn-ghost {
          background: transparent; color:#c8a97e; font-weight:600; font-size:13px;
          border:1.5px solid #c8a97e30; border-radius:10px; padding:10px 20px;
          cursor:pointer; transition:background .2s, border-color .2s;
          font-family:'DM Sans',sans-serif; flex:1;
        }
        .btn-ghost:hover { background:#c8a97e15; border-color:#c8a97e80; }
        .btn-ghost:disabled { opacity:.35; cursor:not-allowed; }
        .btn-outline {
          background: transparent; color:#f0ede8; font-weight:600; font-size:13px;
          border:1.5px solid #2a2a38; border-radius:10px; padding:10px 20px;
          cursor:pointer; transition:background .2s; font-family:'DM Sans',sans-serif; flex:1;
        }
        .btn-outline:hover { background:#1e1e28; }
        textarea {
          width:100%; background:#13131a; border:1.5px solid #2a2a35; border-radius:10px;
          padding:12px 14px; color:#f0ede8; font-size:13px; font-family:'DM Sans',sans-serif;
          resize:none; outline:none; transition:border-color .2s; line-height:1.5;
        }
        textarea:focus { border-color:#c8a97e80; }
        textarea::placeholder { color:#444452; }
        .preview-card {
          background:#13131a; border:1px solid #1e1e26; border-radius:14px;
          overflow:hidden; display:flex; flex-direction:column; align-items:center;
        }
        .preview-img-wrap { width:100%; aspect-ratio:3/4; overflow:hidden; position:relative; background:#0d0d0f; }
        .preview-img-wrap img { width:100%;height:100%;object-fit:cover; }
        .preview-label { padding:10px 14px; font-size:12px; color:#6b6b7a; font-weight:500; width:100%; text-align:center; letter-spacing:.03em; }
        .spinner {
          width:36px;height:36px;border-radius:50%;
          border:3px solid #2a2a38; border-top-color:#c8a97e;
          animation:spin 1s linear infinite;
        }
        @keyframes spin{ to{ transform:rotate(360deg); } }
        .divider { height:1px; background:linear-gradient(90deg,transparent,#2a2a38,transparent); margin:6px 0; }
        .empty-state { width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#2e2e38;font-size:13px; }
        .approved-badge {
          display:inline-flex; align-items:center; gap:5px;
          background:#c8a97e18; border:1px solid #c8a97e40; border-radius:20px;
          padding:3px 10px; font-size:11px; color:#c8a97e; font-weight:600;
          margin-top:6px;
        }
        .error-box {
          background:#3a1a1a; border:1px solid #6b2a2a; border-radius:8px;
          padding:8px 12px; font-size:12px; color:#f09090; margin-top:8px;
          line-height:1.5; word-break:break-word;
        }
        .result-actions {
          position:absolute; top:12px; right:12px; display:flex; gap:8px;
        }
        .result-action-btn {
          background:rgba(12,12,12,0.82); backdrop-filter:blur(8px); border:1px solid #2a2a38;
          color:#f0ede8; border-radius:8px; padding:6px 10px; font-size:11px; font-weight:600;
          text-decoration:none; letter-spacing:.03em; cursor:pointer;
        }
        .result-action-btn:hover { background:rgba(26,26,30,0.92); }
        .result-action-btn:disabled { opacity:.45; cursor:not-allowed; }
        .fullscreen-overlay {
          position:fixed; inset:0; background:rgba(0,0,0,0.92); z-index:9999;
          display:flex; align-items:center; justify-content:center; padding:24px;
        }
        .fullscreen-image {
          max-width:100%; max-height:100%; object-fit:contain;
        }
        .fullscreen-back {
          position:absolute; top:18px; left:18px;
        }
      `}</style>

      {/* ── Header ── */}
      <header style={{ borderBottom: "1px solid #1a1a22", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: "22px", letterSpacing: "-.02em" }}>
          VIRTUAL TRY-ON
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {[1, 2, 3].map((n) => (
            <div key={n} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%",
                background: step >= n ? "linear-gradient(135deg,#c8a97e,#a8865a)" : "#1e1e26",
                border: step >= n ? "none" : "1px solid #2a2a35",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 700, color: step >= n ? "#0d0d0f" : "#4a4a5a",
              }}>{n}</div>
              {n < 3 && <div style={{ width: 20, height: 1, background: step > n ? "#c8a97e40" : "#2a2a35" }} />}
            </div>
          ))}
        </div>
      </header>

      {/* ── Main ── */}
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "25px 20px 40px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>

          {/* ── LEFT COLUMN ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>

            {/* Step 1 – Choose Model */}
            <section>
              <div className="section-tag"><span className="dot" />Step 01</div>
              <div className="section-title">Choose Your Model</div>
              <div className="card-grid" style={{ marginBottom: 10 }}>
                {DATASET.models.map((m) => (
                  <div
                    key={m.id}
                    className={`thumb${personAsset?.id === m.id ? " selected" : ""}`}
                    onClick={() => selectModel(m)}
                  >
                    <img src={m.url} alt={m.label} />
                    <div className="badge">{m.label}</div>
                    <div className="check">✓</div>
                  </div>
                ))}
              </div>
              <div
                className="upload-tile"
                onClick={() => personRef.current?.click()}
                style={{ aspectRatio: "auto", height: 50, flexDirection: "row", gap: 8 }}
              >
                <span style={{ fontSize: 18 }}>↑</span>
                <span>Upload your own photo</span>
                {customPerson && (
                  <span style={{ color: "#c8a97e", marginLeft: "auto", fontSize: 11 }}>✓ Uploaded</span>
                )}
              </div>
            </section>

            <div className="divider" />

            {/* Step 2 – Choose Clothing */}
            <section>
              <div className="section-tag"><span className="dot" />Step 02</div>
              <div className="section-title">Pick a Garment</div>

              <div className="card-grid" style={{ marginBottom: 10 }}>
                {DATASET.clothes.map((c) => (
                  <div
                    key={c.id}
                    className={`thumb${clothAsset?.id === c.id ? " selected" : ""}`}
                    onClick={() => selectCloth(c)}
                  >
                    <img src={c.url} alt={c.label} />
                    <div className="badge">{c.label}</div>
                    <div className="check">✓</div>
                  </div>
                ))}
              </div>

              <div
                className="upload-tile"
                onClick={() => clothRef.current?.click()}
                style={{ aspectRatio: "auto", height: 50, flexDirection: "row", gap: 8 }}
              >
                <span style={{ fontSize: 18 }}>↑</span>
                <span>Upload your own garment</span>
                {customCloth && !approved && (
                  <span style={{ color: "#baa792", marginLeft: "auto", fontSize: 11 }}>✓ Uploaded</span>
                )}
              </div>

              {/* ── AI Generate section ── */}
              <div style={{ marginTop: 14 }}>
                <div style={{ fontSize: 11, color: "#5a5a6a", fontWeight: 600, letterSpacing: ".06em", textTransform: "uppercase", marginBottom: 8 }}>
                  — or generate with AI —
                </div>

                <textarea
                  rows={2}
                  placeholder="e.g. 'A slim-fit navy linen blazer with gold buttons…'"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (!generatingCloth) generateClothing();
                    }
                  }}
                />

                <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                  <button
                    className="btn-ghost"
                    onClick={generateClothing}
                    disabled={generatingCloth || !prompt.trim()}
                  >
                    {generatingCloth
                      ? <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ width: 12, height: 12, borderRadius: "50%", border: "2px solid #c8a97e60", borderTopColor: "#c8a97e", animation: "spin 0.8s linear infinite", display: "inline-block" }} />
                          Generating…
                        </span>
                      : "✦ Generate"
                    }
                  </button>

                  {generatedCloth && !approved && (
                    <>
                      <button className="btn-ghost" onClick={approveCloth}>✓ Use This</button>
                      <button className="btn-outline" onClick={redoGenerate} disabled={generatingCloth}>↺ Redo</button>
                    </>
                  )}

                  {approved && (
                    <button className="btn-outline" onClick={redoGenerate} disabled={generatingCloth}>↺ Regenerate</button>
                  )}
                </div>

                {/* Error message */}
                {genError && (
                  <div className="error-box">⚠ {genError}</div>
                )}

                {/* Approved badge */}
                {approved && (
                  <div className="approved-badge">✓ AI-generated cloth ready for try-on</div>
                )}
              </div>
            </section>
          </div>

          {/* ── RIGHT COLUMN ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

            {/* Preview cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="preview-card">
                <div className="preview-img-wrap">
                  {personSrc
                    ? <img src={personSrc} alt="model" />
                    : <div className="empty-state">No model</div>}
                </div>
                <div className="preview-label">MODEL · {personLabel.toUpperCase()}</div>
              </div>

              <div className="preview-card">
                <div className="preview-img-wrap">
                  {garmentPreviewSrc
                    ? <img src={garmentPreviewSrc} alt="garment" />
                    : <div className="empty-state">No garment</div>}
                </div>
                <div className="preview-label">
                  GARMENT · {activeClothLabel.toUpperCase()}
                </div>
              </div>
            </div>

            {/* Try-On button */}
            <button className="btn-primary" onClick={generatePreview} disabled={!canTryOn}>
              {loading ? "Processing… (1–2 min)" : "✦ Generate Try-On"}
            </button>

            {/* Result panel */}
            <div className="result-panel">
              {loading ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
                  <div className="spinner" />
                  <div style={{ fontSize: 12, color: "#5a5a6a", fontWeight: 500 }}>Running on GPU…</div>
                </div>
              ) : result ? (
                <img src={result} alt="try-on result" className="result-img" />
              ) : (
                <div style={{ textAlign: "center", color: "#2e2e38" }}>
                  <div style={{ fontSize: 48, marginBottom: 12, opacity: .3 }}>◈</div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>Result appears here</div>
                  <div style={{ fontSize: 11, marginTop: 4, color: "#3a3a45" }}>Select model + garment, then try on</div>
                </div>
              )}
              {result && (
                <>
                  <div className="result-actions">
                    <button className="result-action-btn" onClick={() => setIsFullscreen(true)}>
                      ↗ Fullscreen
                    </button>
                  </div>
                  <button
                    className="result-action-btn"
                    onClick={downloadResult}
                    disabled={downloading}
                    style={{ position: "absolute", bottom: 12, right: 12, color: "#c8a97e" }}
                  >
                    {downloading ? "Downloading…" : "↓ Save"}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </main>

      {isFullscreen && result && (
        <div className="fullscreen-overlay">
          <button className="result-action-btn fullscreen-back" onClick={() => setIsFullscreen(false)}>
            ← Back
          </button>
          <img src={result} alt="try-on full screen" className="fullscreen-image" />
        </div>
      )}

      {/* Hidden file inputs */}
      <input ref={personRef} type="file" hidden accept="image/*" onChange={(e) => handleUpload(e, "person")} />
      <input ref={clothRef}  type="file" hidden accept="image/*" onChange={(e) => handleUpload(e, "cloth")}  />
    </div>
  );
}

