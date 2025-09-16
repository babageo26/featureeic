
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import time
import cv2
import numpy as np

from .viz import draw_matches
from .io import save_outputs

def load_color(src):
    if isinstance(src, str):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {src}")
        return img
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        if src.ndim == 3:
            return src.copy()
    raise TypeError("img must be path or numpy array")

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize_to(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def _unsharp_mask(gray: np.ndarray, ksize: int = 0, amount: float = 1.5) -> np.ndarray:
    if ksize <= 0:
        k = max(3, (min(gray.shape[:2]) // 100) * 2 + 1)
    else:
        k = ksize if ksize % 2 == 1 else ksize + 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def preprocess(gray: np.ndarray,
               do_clahe: bool = False,
               do_unsharp: bool = False,
               do_denoise: bool = False,
               strong_unsharp: bool = False,
               clip_limit: float = 2.0,
               tile_grid_size: Tuple[int,int] = (8,8)) -> np.ndarray:
    out = gray
    if do_denoise:
        out = cv2.fastNlMeansDenoising(out, None, h=7, templateWindowSize=7, searchWindowSize=21)
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out = clahe.apply(out)
    if do_unsharp:
        out = _unsharp_mask(out, amount=2.0 if strong_unsharp else 1.2)
    return out

def build_extractor(algo: str):
    algo_u = algo.upper()
    if algo_u == "SIFT":
        return cv2.SIFT_create()
    if algo_u == "ORB":
        return cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8,
                              edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    if algo_u == "BRIEF":
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        try:
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)
        except Exception as e:
            raise RuntimeError("BRIEF requires opencv-contrib-python.") from e
        return (fast, brief)
    raise ValueError(f"Unknown algo: {algo}")

def detect_and_describe(algo: str, imgA: np.ndarray, imgB: np.ndarray):
    algo_u = algo.upper()
    extractor = build_extractor(algo_u)
    t0 = time.perf_counter()
    if algo_u == "BRIEF":
        fast, brief = extractor
        kp1 = fast.detect(imgA, None)
        kp2 = fast.detect(imgB, None)
        kp1, des1 = brief.compute(imgA, kp1)
        kp2, des2 = brief.compute(imgB, kp2)
    else:
        kp1, des1 = extractor.detectAndCompute(imgA, None)
        kp2, des2 = extractor.detectAndCompute(imgB, None)
    t1 = time.perf_counter()
    meta = {"t_detect_s": t1 - t0}
    return kp1, des1, kp2, des2, meta

def match_with_ratio(algo: str, des1, des2, ratio: float = 0.75):
    algo_u = algo.upper()
    if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
        return [], {"t_match_s": 0.0}
    norm = cv2.NORM_L2 if algo_u == "SIFT" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    t0 = time.perf_counter()
    try:
        raw = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        bf2 = cv2.BFMatcher(norm, crossCheck=True)
        raw = [[m] for m in bf2.match(des1, des2)]
    good = []
    for pair in raw:
        if len(pair) == 1:
            good.append(pair[0])
        else:
            m, n = pair[0], pair[1]
            if n.distance == 0:
                if m.distance < 0.8:
                    good.append(m)
            elif m.distance < ratio * n.distance:
                good.append(m)
    t1 = time.perf_counter()
    return good, {"t_match_s": t1 - t0}

def find_homography_ransac(kp1, kp2, matches, thresh: float = 3.0):
    if len(matches) < 4:
        return None, None, {"t_ransac_s": 0.0}
    t0 = time.perf_counter()
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=2000, confidence=0.995)
    t1 = time.perf_counter()
    inlier_mask = mask.ravel().astype(bool) if mask is not None else None
    return H, inlier_mask, {"t_ransac_s": t1 - t0}

@dataclass
class RunConfig:
    algo: str
    ratio: float = 0.75
    ransac_thresh: float = 3.0
    use_preprocess: bool = True
    pp_kwargs: Dict[str, Any] = None

def run_once(
    algo: str,
    imgA,
    imgB,
    pp_kwargs: Dict[str, Any] | None = None,
    use_preprocess: bool = True,
    ratio: float = 0.75,
    ransac_thresh: float = 3.0,
    draw: bool = True,
    save: bool = False,
    save_dir: Optional[str] = None,
    label: Optional[str] = None,
    # compatibility params (ignored/optional)
    preset_override: Optional[str] = None,
    max_draw: Optional[int] = None,
):
    pp_kwargs = pp_kwargs or {}
    t_total0 = time.perf_counter()

    imgA_color = load_color(imgA)
    imgB_color = load_color(imgB)
    imgB_color = resize_to(imgB_color, imgA_color)
    imgA_gray = to_gray(imgA_color)
    imgB_gray = to_gray(imgB_color)

    if use_preprocess:
        imgA_proc = preprocess(imgA_gray, **pp_kwargs)
        imgB_proc = preprocess(imgB_gray, **pp_kwargs)
        preprocess_used = True
    else:
        imgA_proc, imgB_proc = imgA_gray, imgB_gray
        preprocess_used = False

    kp1, des1, kp2, des2, meta_det = detect_and_describe(algo, imgA_proc, imgB_proc)

    if algo.upper() == "SIFT":
        r = ratio if ratio is not None else 0.75
    elif algo.upper() == "ORB":
        r = ratio if ratio is not None else 0.85
    else:
        r = ratio if ratio is not None else 0.95
    matches, meta_match = match_with_ratio(algo, des1, des2, ratio=r)

    h, w = imgA_gray.shape[:2]
    thr = ransac_thresh if max(h, w) <= 1600 else max(4.0, ransac_thresh)
    H, inlier_mask, meta_ransac = find_homography_ransac(kp1, kp2, matches, thresh=thr)

    vis = None
    if draw:
        vis = draw_matches(imgA_color, kp1, imgB_color, kp2, matches, inlier_mask, max_draw=max_draw)

    n_kp1 = len(kp1) if kp1 is not None else 0
    n_kp2 = len(kp2) if kp2 is not None else 0
    n_match = len(matches)
    n_inlier = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = (n_inlier / max(n_match, 1)) if n_match > 0 else 0.0

    t_total1 = time.perf_counter()
    result = {
        "algo": algo.upper(),
        "label": label or "",
        "n_kp1": n_kp1, "n_kp2": n_kp2,
        "n_match": n_match, "n_inlier": n_inlier,
        "inlier_ratio": float(inlier_ratio),
        "t_detect_s": float(meta_det.get("t_detect_s", 0.0)),
        "t_match_s": float(meta_match.get("t_match_s", 0.0)),
        "t_ransac_s": float(meta_ransac.get("t_ransac_s", 0.0)),
        "t_total_s": float(t_total1 - t_total0),
        "preprocess_enabled": preprocess_used,
    }

    if save and save_dir is not None:
        save_outputs(save_dir, result, vis)

    return result, vis
