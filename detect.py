from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class CandidateRegion:
    contour: np.ndarray
    polygon: List[Tuple[float, float]]
    area_px: float
    perimeter_px: float


@dataclass
class DetectionDebug:
    gray: np.ndarray
    binary: np.ndarray
    thickness_mask: np.ndarray
    closed_mask: np.ndarray
    contour_preview: np.ndarray
    stats: Dict[str, str]


def _kernel_from_preset(preset: str) -> int:
    preset_map = {
        "Light": 3,
        "Medium": 5,
        "Heavy": 7,
    }
    return preset_map.get(preset, 5)


def detect_regions(
    gray: np.ndarray,
    thickness_preset: str,
    gap_close: str,
    min_area_sqft: float,
    ft_per_pixel: float,
    dpi: int,
) -> Tuple[List[CandidateRegion], DetectionDebug]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_size = _kernel_from_preset(thickness_preset)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    thickness_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if gap_close == "Low":
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_mask = cv2.morphologyEx(thickness_mask, cv2.MORPH_CLOSE, close_kernel)
    else:
        closed_mask = thickness_mask.copy()

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area_px = min_area_sqft / (ft_per_pixel ** 2)
    candidates: List[CandidateRegion] = []

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < min_area_px:
            continue
        perimeter_px = cv2.arcLength(contour, True)
        epsilon = 0.002 * perimeter_px
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        candidates.append(
            CandidateRegion(
                contour=contour,
                polygon=polygon,
                area_px=area_px,
                perimeter_px=perimeter_px,
            )
        )

    contour_preview = cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_preview, [c.contour for c in candidates], -1, (0, 255, 0), 2)

    stats = {
        "dpi": str(dpi),
        "kernel_size": str(kernel_size),
        "contours_found": str(len(contours)),
        "candidates": str(len(candidates)),
        "min_area_sqft": f"{min_area_sqft:.2f}",
    }

    debug = DetectionDebug(
        gray=gray,
        binary=binary,
        thickness_mask=thickness_mask,
        closed_mask=closed_mask,
        contour_preview=contour_preview,
        stats=stats,
    )
    return candidates, debug
