
import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional
import cv2

# --- helpers ---
def _safe_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# --- unified output writer (baru) ---
def save_outputs(save_dir: str, result: Dict[str, Any], vis_img):
    """
    Menyimpan visual (jpg) dan meng-append satu baris ringkasan ke CSV:
      runs/<algo>/vis/<label>_<algo>.jpg
      runs/<algo>/summary_<algo>.csv
    Termasuk kolom wide: <ALGO> MATCHES / <ALGO> MATCHES TEST / <ALGO> SPEED (ms)
    """
    algo = result.get("algo", "ALGO").lower()
    base = Path(save_dir)
    vis_dir = _safe_dir(base / "vis")
    csv_dir = _safe_dir(base)

    # Save visualization
    if vis_img is not None:
        safe_label = (result.get("label") or "result").replace("/", "_")
        vis_path = vis_dir / f"{safe_label}_{algo}.jpg"
        cv2.imwrite(str(vis_path), vis_img)

    # Prepare CSV path (per-algo summary file)
    csv_path = csv_dir / f"summary_{algo}.csv"
    # Append row with both raw metrics and a 'wide' style cols
    fieldnames = [
        "label", "algo", "n_kp1", "n_kp2", "n_match", "n_inlier", "inlier_ratio",
        "t_detect_s", "t_match_s", "t_ransac_s", "t_total_s", "preprocess_enabled",
        f"{algo.upper()} MATCHES", f"{algo.upper()} MATCHES TEST", f"{algo.upper()} SPEED (ms)"
    ]
    # Ensure file exists with header
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Build row
    row = {k: result.get(k, "") for k in [
        "label","algo","n_kp1","n_kp2","n_match","n_inlier","inlier_ratio",
        "t_detect_s","t_match_s","t_ransac_s","t_total_s","preprocess_enabled"
    ]}
    # Wide-style values
    row[f"{algo.upper()} MATCHES"] = result.get("n_match", 0)
    row[f"{algo.upper()} MATCHES TEST"] = result.get("n_inlier", 0)
    row[f"{algo.upper()} SPEED (ms)"] = int(float(result.get("t_total_s", 0.0)) * 1000)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

# --- kompatibilitas mundur (lama) ---
def ensure_dirs(path) -> Path:
    """Compatibility wrapper: ensure directory exists and return Path."""
    return _safe_dir(Path(path))

def save_visual(vis_path, vis_img) -> None:
    """Compatibility wrapper: save visualization image to a given path."""
    cv2.imwrite(str(vis_path), vis_img)

def append_summary_row(csv_path, row: Dict[str, Any], fieldnames: Optional[List[str]] = None) -> None:
    """
    Compatibility wrapper: append a summary row to a CSV file.
    If fieldnames not provided, infer from row keys (order not guaranteed).
    """
    csv_path = Path(csv_path)
    exists = csv_path.exists()
    if fieldnames is None:
        fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
