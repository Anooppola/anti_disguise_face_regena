"""
frontend/streamlit_app.py - Anti-Disguise Face Reconstruction Frontend
"""

import io
import os

import requests
import streamlit as st
from PIL import Image

# ─── Configuration ─────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Anti-Disguise Face Reconstruction",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Background + global typography */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero banner */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }

    /* Metric pill */
    .metric-pill {
        display: inline-block;
        background: rgba(96,165,250,0.15);
        border: 1px solid rgba(96,165,250,0.3);
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        font-size: 0.85rem;
        color: #93c5fd;
        margin: 0.2rem;
    }

    /* Status badges */
    .badge-ok   { color: #34d399; font-weight: 600; }
    .badge-warn { color: #fbbf24; font-weight: 600; }
    .badge-err  { color: #f87171; font-weight: 600; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(167,139,250,0.4) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.03) !important;
        padding: 1rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Image captions */
    .img-label {
        text-align: center;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    hr { border-color: rgba(255,255,255,0.08); }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helper functions ──────────────────────────────────────────────────────────

def check_api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        return r.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def call_predict_api(image_bytes: bytes) -> bytes | None:
    """POST image bytes to /predict, return PNG response bytes or None."""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": ("input.png", image_bytes, "image/png")},
            timeout=60,
        )
        r.raise_for_status()
        return r.content
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ API error: {exc}")
        return None


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎭 Anti-Disguise GAN")
    st.markdown("---")

    # API status
    health = check_api_health()
    if health.get("status") == "ok":
        model_ok = health.get("model_loaded", False)
        st.markdown(f'<span class="badge-ok">● API Online</span>', unsafe_allow_html=True)
        status_txt = "Model Loaded ✅" if model_ok else "Model NOT loaded ⚠️"
        color = "badge-ok" if model_ok else "badge-warn"
        st.markdown(f'<span class="{color}">{status_txt}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">● API Offline</span>', unsafe_allow_html=True)
        st.caption(f"Could not reach {API_URL}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
- **Architecture**: Pix2Pix GAN
- **Generator**: U-Net (256×256)
- **Discriminator**: PatchGAN 70×70
- **Losses**: Adversarial + L1 + VGG19
- **API**: FastAPI → port **8000**
- **Tracking**: MLflow → port **5000**
""")

    st.markdown("---")
    st.markdown("### 🔧 Advanced")
    api_url_input = st.text_input("Backend URL", value=API_URL)
    if api_url_input != API_URL:
        API_URL = api_url_input

    st.markdown("---")
    st.markdown("**Quick links**")
    st.markdown(f"[📊 MLflow UI](http://localhost:5000)")
    st.markdown(f"[📖 API Docs]({API_URL}/docs)")


# ─── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🎭 Anti-Disguise Face Reconstruction</h1>
    <p>Upload a masked or occluded face image and let the Pix2Pix GAN reconstruct the original.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="margin-bottom:2rem">', unsafe_allow_html=True)

# ─── Main layout ───────────────────────────────────────────────────────────────

upload_col, result_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Masked Face")
    uploaded = st.file_uploader(
        "Drag & drop or click to select",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a masked or occluded face image (256×256 works best)",
    )

    if uploaded:
        img_bytes = uploaded.read()
        input_img = bytes_to_pil(img_bytes)
        st.image(input_img, use_container_width=True)
        st.markdown('<p class="img-label">Input (Masked)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="metric-pill">Size: {input_img.size[0]}×{input_img.size[1]}</span>'
            f'<span class="metric-pill">Mode: {input_img.mode}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        st.markdown("")
        run_btn = st.button("🚀 Reconstruct Face", use_container_width=True)
    else:
        run_btn = False

with result_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🖼️ Reconstructed Face")

    if not uploaded:
        st.markdown("""
<div style="height:300px;display:flex;align-items:center;justify-content:center;
            color:#4b5563;flex-direction:column;gap:1rem;">
    <span style="font-size:4rem;">🎭</span>
    <span>Upload an image to get started</span>
</div>
""", unsafe_allow_html=True)

    elif run_btn:
        with st.spinner("Running GAN inference…"):
            result_bytes = call_predict_api(pil_to_bytes(input_img))

        if result_bytes:
            result_img = bytes_to_pil(result_bytes)
            st.image(result_img, use_container_width=True)
            st.markdown('<p class="img-label">Output (Reconstructed)</p>', unsafe_allow_html=True)
            st.success("✅ Reconstruction complete!")

            st.download_button(
                "⬇️ Download Result",
                data=result_bytes,
                file_name="reconstructed_face.png",
                mime="image/png",
                use_container_width=True,
            )

            # ── Side-by-side comparison ──
            st.markdown("---")
            st.markdown("#### Before vs After")
            c1, c2 = st.columns(2)
            with c1:
                st.image(input_img, use_container_width=True)
                st.markdown('<p class="img-label">Masked</p>', unsafe_allow_html=True)
            with c2:
                st.image(result_img, use_container_width=True)
                st.markdown('<p class="img-label">Reconstructed</p>', unsafe_allow_html=True)
        else:
            st.error("Inference failed. Check that the API and model are running.")

    else:
        st.markdown("""
<div style="height:300px;display:flex;align-items:center;justify-content:center;
            color:#4b5563;flex-direction:column;gap:1rem;">
    <span style="font-size:3rem;">⬅️</span>
    <span>Click "Reconstruct Face" to run inference</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── How-to section ────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖 How to use this app"):
    st.markdown("""
1. **Upload** a masked or occluded face image using the uploader on the left.
2. Click **🚀 Reconstruct Face** to send the image to the GAN backend.
3. The **reconstructed face** appears on the right.
4. Use **⬇️ Download Result** to save the output.
5. View training metrics on [MLflow UI](http://localhost:5000).

**API curl example:**
```bash
curl -X POST http://localhost:8000/predict \\
     -F "file=@masked_face.png" \\
     --output reconstructed.png
```
""")
