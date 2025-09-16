
from typing import Optional
import numpy as np
import cv2

def _stack_side_by_side(imgA, imgB):
    h1, w1 = imgA.shape[:2]
    h2, w2 = imgB.shape[:2]
    H = max(h1, h2)
    s1 = cv2.resize(imgA, (w1, H))
    s2 = cv2.resize(imgB, (w2, H))
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:H, :w1] = s1
    canvas[:H, w1:w1+w2] = s2
    return canvas, w1

def draw_matches(imgA_color, kp1, imgB_color, kp2, matches, inlier_mask: Optional[np.ndarray] = None, max_draw: int | None = None):
    canvas, offset = _stack_side_by_side(imgA_color, imgB_color)
    if matches is None:
        return canvas
    limit = len(matches) if max_draw is None else min(len(matches), max_draw)
    for i, m in enumerate(matches[:limit]):
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        pt2_shift = (int(pt2[0] + offset), int(pt2[1]))
        is_in = (inlier_mask[i] == 1) if (inlier_mask is not None and i < len(inlier_mask)) else False
        color = (40, 220, 40) if is_in else (150, 150, 150)
        cv2.line(canvas, pt1, pt2_shift, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2_shift, 2, color, -1, cv2.LINE_AA)
    return canvas
