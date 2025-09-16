
import os, sys

# === Import path guard for local modules (fx.*) ===
from pathlib import Path as _PathGuard
import sys as _sys_guard
_ROOT = _PathGuard(__file__).parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in _sys_guard.path:
    _sys_guard.path.insert(0, str(_SRC))
# === /Import path guard ===
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import streamlit as st
import pandas as pd
from pathlib import Path

# Import pipeline + preprocess from our package
from fx.pipeline import run_once, preprocess

st.set_page_config(page_title="Image Feature Matching — SIFT / ORB / BRIEF", page_icon="🧩", layout="wide")

# ---- Session state: store accumulated rows per algorithm ----
if "wide_rows" not in st.session_state:
    st.session_state.wide_rows = {"SIFT": [], "ORB": [], "BRIEF": []}

# ================= Sidebar =================
st.sidebar.header("Pengaturan")
algo = st.sidebar.selectbox("Pilih algoritma", ["SIFT", "ORB", "BRIEF"], index=0)

preset_label = st.sidebar.selectbox(
    "Preset preprocessing (opsional)",
    ["AUTO", "PP_DEFAULT", "PP_BLUR", "PP_BRIEF", "TANPA PREPROCESS"],
    index=0
)
st.sidebar.caption("AUTO: SIFT/ORB→pakai CLAHE + Unsharp Mask, BRIEF → Denois")
st.sidebar.caption("PP_DEFAULT: CLAHE + Unsharp Mask")
st.sidebar.caption("PP_BLUR: Strong_Unsharp")
st.sidebar.caption("PP_BRIEF: Denoise")
st.sidebar.caption("TANPA PREPROCESS: Tanpa Enhacement")

ratio_defaults = {"SIFT": 0.75, "ORB": 0.85, "BRIEF": 0.95}
ratio = st.sidebar.slider("Lowe Ratio", min_value=0.5, max_value=0.99, value=ratio_defaults[algo], step=0.01)
ransac_thresh = st.sidebar.slider("Ambang RANSAC (px)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

# Opsi limit garis visual (opsional) — supaya kompatibel dengan max_draw
limit_draw = st.sidebar.checkbox("Batasi jumlah garis visual", value=False)
max_draw = st.sidebar.slider("Maks garis yang digambar", 50, 2000, 400, 50) if limit_draw else None

st.sidebar.caption("Lowe Ratio: Semakin Ratio Kecil semakin ketat pencocokan match(Andal,sedikit match)")
st.sidebar.caption("Ambang Ransac: Semakin kecil nilai px maka semakin ketat, hanya match sangat presisi yang dianggap inlier")

# ================= Main =================
st.markdown("<h1 style='text-align:center'>🧩 Image Feature Matching — SIFT / ORB / BRIEF</h1>", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    st.subheader("Upload Gambar A (basis / original)")
    fileA = st.file_uploader("Drag and drop file here", type=["jpg","jpeg","png"], key="fileA")
with colB:
    st.subheader("Upload Gambar B (varian / target)")
    fileB = st.file_uploader("Drag and drop file here", type=["jpg","jpeg","png"], key="fileB")

label = st.text_input("Label CONDITION / TEST", placeholder="mis. rotate / scale / oblique1 / custom_A_vs_B")

# Histogram controls
hist_req = st.checkbox("📈 Tampilkan histogram sebelum & sesudah preprocessing", value=False, help="Bantu melihat pengaruh enhancement.")
show_hist_btn = st.button("Cek Histogram", use_container_width=True) if hist_req else None

# Run button
run_btn = st.button("🚀 Jalankan Pengujian", use_container_width=True, type="primary")

def _read_cv2(file):
    data = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# Map preset -> use_preprocess & pp_kwargs
use_preprocess = True
pp_kwargs = {}
if preset_label == "TANPA PREPROCESS":
    use_preprocess = False
    pp_kwargs = {}
elif preset_label == "AUTO (disarankan)":
    if algo in ("SIFT","ORB"):
        pp_kwargs = dict(do_clahe=True, do_unsharp=True)
    else:  # BRIEF
        pp_kwargs = dict(do_denoise=True)
elif preset_label == "PP_DEFAULT":
    pp_kwargs = dict(do_clahe=True, do_unsharp=True)
elif preset_label == "PP_BLUR":
    pp_kwargs = dict(do_clahe=True, do_unsharp=True, strong_unsharp=True)
elif preset_label == "PP_BRIEF":
    pp_kwargs = dict(do_denoise=True)

# Helpers for histogram preview
def _prep_for_hist(imgA_bgr, imgB_bgr, use_preprocess: bool, pp_kwargs: dict):
    imgB_bgr_rs = cv2.resize(imgB_bgr, (imgA_bgr.shape[1], imgA_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    A_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(imgB_bgr_rs, cv2.COLOR_BGR2GRAY)
    if use_preprocess:
        A_post = preprocess(A_gray, **pp_kwargs)
        B_post = preprocess(B_gray, **pp_kwargs)
    else:
        A_post, B_post = A_gray, B_gray
    return A_gray, B_gray, A_post, B_post

def _plot_hist(gray_img, title):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    hist, bins = np.histogram(gray_img.flatten(), bins=256, range=[0,256])
    plt.plot(hist)  # default color
    plt.title(title)
    plt.xlabel("Intensitas (0-255)")
    plt.ylabel("Frekuensi")
    return fig

# ---- Histogram preview ----
if hist_req and show_hist_btn:
    if fileA is None or fileB is None:
        st.error("Unggah kedua gambar terlebih dahulu untuk melihat histogram.")
    else:
        imgA0 = _read_cv2(fileA)
        imgB0 = _read_cv2(fileB)
        A_gray, B_gray, A_post, B_post = _prep_for_hist(imgA0, imgB0, use_preprocess, pp_kwargs)

        cH1, cH2 = st.columns(2)
        with cH1:
            st.subheader("Histogram A (sebelum)")
            st.pyplot(_plot_hist(A_gray, "A - sebelum"))
        with cH2:
            st.subheader("Histogram A (sesudah)")
            st.pyplot(_plot_hist(A_post, "A - sesudah"))

        cH3, cH4 = st.columns(2)
        with cH3:
            st.subheader("Histogram B (sebelum)")
            st.pyplot(_plot_hist(B_gray, "B - sebelum"))
        with cH4:
            st.subheader("Histogram B (sesudah)")
            st.pyplot(_plot_hist(B_post, "B - sesudah"))

# Output base dir
out_base = Path("runs") / algo.lower()
out_base.mkdir(parents=True, exist_ok=True)

# ---- Run main pipeline ----
if run_btn:
    if fileA is None or fileB is None:
        st.error("Mohon unggah kedua gambar terlebih dahulu.")
    else:
        imgA = _read_cv2(fileA)
        imgB = _read_cv2(fileB)
        with st.spinner("Memproses..."):
            result, vis = run_once(
                algo=algo,
                imgA=imgA, imgB=imgB,
                pp_kwargs=pp_kwargs,
                use_preprocess=use_preprocess,
                ratio=ratio,
                ransac_thresh=ransac_thresh,
                draw=True,
                save=True,
                save_dir=str(out_base),
                label=label,
                # Compatibility args (aman jika tidak dipakai di backend)
                preset_override=preset_label,
                max_draw=max_draw,
            )
        # Show outputs
        st.success("Selesai!")
        c1, c2 = st.columns([2,1])
        with c1:
            if vis is not None:
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Korespondensi (hijau = inlier)")

            # === TABEL DI KOLOM KIRI (di bawah visual) ===
            algo_u = result["algo"].upper()
            row = {
                "CONDITION": result.get("label", ""),
                "ALGO": algo_u,
                f"{algo_u} MATCHES": result.get("n_match", 0),
                f"{algo_u} MATCHES TEST": result.get("n_inlier", 0),
                f"{algo_u} SPEED (ms)": round(float(result.get("t_total_s", 0.0)) * 1000, 1),
            }
            df_summary = pd.DataFrame([row])
            st.markdown("### Tabel Ringkas")
            st.dataframe(df_summary, use_container_width=True)

            st.markdown("### Tabel Akumulasi (per algoritma)")
            st.session_state.wide_rows[algo_u].append(row)
            df_all = pd.DataFrame(st.session_state.wide_rows[algo_u])
            st.dataframe(df_all, use_container_width=True)
            st.download_button(
                "Unduh Tabel Akumulasi (CSV)",
                data=df_all.to_csv(index=False).encode("utf-8"),
                file_name=f"summary_wide_{algo_u.lower()}.csv",
                mime="text/csv"
            )

        with c2:
            st.markdown("### Metrik")
            st.write({
                "algo": result["algo"],
                "label": result["label"],
                "n_kp1": result["n_kp1"],
                "n_kp2": result["n_kp2"],
                "n_match": result["n_match"],
                "n_inlier": result["n_inlier"],
                "inlier_ratio": round(result["inlier_ratio"], 3),
                "t_detect_s": round(result["t_detect_s"], 4),
                "t_match_s": round(result["t_match_s"], 4),
                "t_ransac_s": round(result["t_ransac_s"], 4),
                "t_total_s": round(result["t_total_s"], 4),
                "preprocess_enabled": result["preprocess_enabled"],
            })
            st.caption(f"Preprocessing: **{'Aktif' if result.get('preprocess_enabled') else 'Nonaktif'}**")

        st.info(f"Hasil & CSV ditulis ke folder: `{out_base}` — summary: `summary_{algo.lower()}.csv`")



# === Export helpers (robust) ===
# Fitur export aman + fallback: bila tidak ada yang teregistrasi di session, kita scan globals()
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import io, os
    from pathlib import Path
    try:
        import cv2
    except Exception:
        cv2 = None

    ROOT = Path(__file__).parent
    RUNS_DIR = ROOT / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if "exports" not in st.session_state:
        st.session_state["exports"] = {"dfs": {}, "imgs": {}}

    def register_df(name: str, df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.session_state["exports"]["dfs"][name] = df

    def register_img(name: str, img):
        try:
            from PIL import Image as _PILImage
        except Exception:
            _PILImage = None

        if _PILImage and isinstance(img, _PILImage.Image):
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            st.session_state["exports"]["imgs"][name] = arr
            return

        if isinstance(img, np.ndarray) and img.size > 0:
            arr = img
            if arr.dtype.kind == "f":
                arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            st.session_state["exports"]["imgs"][name] = arr

    def _collect_dataframes(namespace):
        return {k:v for k,v in namespace.items()
                if (not k.startswith("_"))
                and isinstance(v, pd.DataFrame)
                and not v.empty}

    def _collect_images(namespace):
        return {k:v for k,v in namespace.items()
                if (not k.startswith("_"))
                and isinstance(v, np.ndarray)
                and v.size > 0}

    def _to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode()

    def _to_png_bytes(img: "np.ndarray") -> bytes:
        if cv2 is None:
            from PIL import Image
            import io as _io
            if img.ndim == 2:
                pil = Image.fromarray(img)
            elif img.ndim == 3 and img.shape[2] == 3:
                pil = Image.fromarray(img)
            else:
                raise RuntimeError("Format gambar tidak dikenali untuk export.")
            buf = _io.BytesIO()
            pil.save(buf, format="PNG")
            return buf.getvalue()
        arr = img
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("Gagal meng-encode PNG (periksa dtype dan range piksel).")
        return bytes(buf)

    def export_sidebar():
        with st.sidebar:
            st.markdown("### 📤 Export")
            st.caption("""Hasil terakhir tersimpan di session. Jika kosong, sistem akan mendeteksi otomatis dari variabel global.""")

Jika kosong, sistem akan mendeteksi otomatis dari variabel global.")

            # Primary: use session-registered objects
            dfs = dict(st.session_state["exports"]["dfs"])
            imgs = dict(st.session_state["exports"]["imgs"])

            # Fallback: scan globals if nothing registered
            if not dfs and not imgs:
                g = globals()
                dfs = _collect_dataframes(g)
                imgs = _collect_images(g)
                if dfs or imgs:
                    st.info("Menggunakan fallback: mendeteksi objek dari globals().")

            if dfs:
                names = sorted(dfs.keys())
                picked = st.selectbox("Pilih DataFrame (CSV)", names, key="exp_pick_df")
                if picked:
                    csvb = _to_csv_bytes(dfs[picked])
                    st.download_button("Download CSV", csvb, file_name=f"{picked}.csv", mime="text/csv", key="exp_btn_df")
                    if st.checkbox("Simpan ke runs/", key="exp_save_df"):
                        (RUNS_DIR / f"{picked}.csv").write_bytes(csvb)
                        st.success(f"Saved: runs/{picked}.csv")

            if imgs:
                names_i = sorted(imgs.keys())
                picked_i = st.selectbox("Pilih Gambar (PNG)", names_i, key="exp_pick_img")
                if picked_i:
                    try:
                        pnb = _to_png_bytes(imgs[picked_i])
                        st.download_button("Download PNG", pnb, file_name=f"{picked_i}.png", mime="image/png", key="exp_btn_img")
                        if st.checkbox("Simpan gambar ke runs/", key="exp_save_img"):
                            (RUNS_DIR / f"{picked_i}.png").write_bytes(pnb)
                            st.success(f"Saved: runs/{picked_i}.png")
                    except Exception as e:
                        st.warning(f"Gagal menyiapkan unduhan gambar: {e}")

    export_sidebar()
except Exception:
    pass
# === /Export helpers (robust) ===




# === Optional: auto-register common variable names if they exist ===
try:
    # Daftar variabel yang umum dipakai di pipeline Anda
    for _name in ["summary_sift", "summary_orb", "summary_brief", "summary_df", "results_df"]:
        if _name in globals():
            register_df(_name, globals()[_name])
    for _name in ["vis", "matches_img", "vis_sift", "vis_orb", "vis_brief"]:
        if _name in globals():
            register_img(_name, globals()[_name])
except Exception:
    pass
