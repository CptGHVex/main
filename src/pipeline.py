"""Bid-safe plan takeoff pipeline.

Pipeline steps:
1) Render vector PDF to grayscale raster at target DPI.
2) Adaptive threshold to binarize ink vs. paper.
3) Thickness gate via morphological open (+ optional close).
4) Find closed external contours.
5) Filter contours with bid-safe heuristics (area, vertex count, sanity score).
6) Convert contours to polygons and compute geometry with calibration.
7) Export accepted polygons to CSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import csv

import cv2
import fitz  # PyMuPDF
import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class Calibration:
    """Pixel-to-feet calibration derived from two clicks on the plan.

    Attributes:
        pixel_distance: distance between the two calibration clicks in pixels.
        real_distance_feet: real-world distance between the clicks in feet.
    """

    pixel_distance: float
    real_distance_feet: float

    @property
    def feet_per_pixel(self) -> float:
        if self.pixel_distance <= 0:
            raise ValueError("pixel_distance must be > 0")
        if self.real_distance_feet <= 0:
            raise ValueError("real_distance_feet must be > 0")
        return self.real_distance_feet / self.pixel_distance


@dataclass(frozen=True)
class CandidateMetrics:
    contour_index: int
    area_sqft: float
    perimeter_lf: float
    vertex_count: int
    sanity_score: float
    flagged: bool


@dataclass(frozen=True)
class PipelineConfig:
    dpi: int = 600
    adaptive_block_size: int = 35
    adaptive_c: int = 10
    line_weight_threshold_px: int = 5
    close_kernel_px: Optional[int] = 3
    min_area_sqft: float = 1.0
    max_vertex_count: int = 1000
    sanity_ratio_threshold: float = 0.8


@dataclass(frozen=True)
class PipelineResult:
    polygons: List[Polygon]
    metrics: List[CandidateMetrics]


def render_pdf_to_grayscale(pdf_path: Path, page_number: int, dpi: int) -> np.ndarray:
    """Render a vector PDF page to a grayscale raster image."""
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    finally:
        doc.close()


def adaptive_binarize(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    """Adaptive threshold into ink vs paper (white background)."""
    if block_size % 2 == 0:
        raise ValueError("adaptive_block_size must be odd")
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def thickness_gate(
    binary: np.ndarray,
    line_weight_threshold_px: int,
    close_kernel_px: Optional[int],
) -> np.ndarray:
    """Remove thin lines while preserving heavy outlines."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (line_weight_threshold_px, line_weight_threshold_px),
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if close_kernel_px:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_kernel_px, close_kernel_px),
        )
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return opened


def find_closed_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Find closed external contours from the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contour_to_polygon(contour: np.ndarray) -> Polygon:
    """Convert an OpenCV contour into a Shapely polygon."""
    if contour.ndim != 3 or contour.shape[1] != 1:
        raise ValueError("Unexpected contour shape")
    points = contour[:, 0, :].astype(float)
    return Polygon(points)


def compute_geometry(polygon: Polygon, calibration: Calibration) -> Tuple[float, float]:
    """Compute area (sqft) and perimeter (linear feet)."""
    feet_per_pixel = calibration.feet_per_pixel
    area_sqft = polygon.area * (feet_per_pixel**2)
    perimeter_lf = polygon.length * feet_per_pixel
    return area_sqft, perimeter_lf


def sanity_score(polygon: Polygon) -> float:
    """Perimeter/area ratio for sanity checks (lower is more compact)."""
    if polygon.area == 0:
        return float("inf")
    return polygon.length / polygon.area


def filter_candidates(
    polygons: Sequence[Polygon],
    calibration: Calibration,
    config: PipelineConfig,
) -> PipelineResult:
    metrics: List[CandidateMetrics] = []
    accepted: List[Polygon] = []

    for idx, polygon in enumerate(polygons):
        if polygon.is_empty:
            continue
        area_sqft, perimeter_lf = compute_geometry(polygon, calibration)
        vertices = len(polygon.exterior.coords)
        score = sanity_score(polygon)
        flagged = score > config.sanity_ratio_threshold
        if area_sqft < config.min_area_sqft:
            continue
        if vertices > config.max_vertex_count:
            continue
        metrics.append(
            CandidateMetrics(
                contour_index=idx,
                area_sqft=area_sqft,
                perimeter_lf=perimeter_lf,
                vertex_count=vertices,
                sanity_score=score,
                flagged=flagged,
            )
        )
        accepted.append(polygon)

    return PipelineResult(polygons=accepted, metrics=metrics)


def export_to_csv(metrics: Iterable[CandidateMetrics], output_path: Path) -> None:
    """Export accepted polygons to a CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "contour_index",
                "area_sqft",
                "perimeter_lf",
                "vertex_count",
                "sanity_score",
                "flagged",
            ],
        )
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "contour_index": item.contour_index,
                    "area_sqft": item.area_sqft,
                    "perimeter_lf": item.perimeter_lf,
                    "vertex_count": item.vertex_count,
                    "sanity_score": item.sanity_score,
                    "flagged": item.flagged,
                }
            )


def run_pipeline(
    pdf_path: Path,
    page_number: int,
    calibration: Calibration,
    config: PipelineConfig,
) -> PipelineResult:
    """Run the full pipeline from PDF to filtered polygons."""
    gray = render_pdf_to_grayscale(pdf_path, page_number, config.dpi)
    binary = adaptive_binarize(gray, config.adaptive_block_size, config.adaptive_c)
    mask = thickness_gate(binary, config.line_weight_threshold_px, config.close_kernel_px)
    contours = find_closed_contours(mask)
    polygons = [contour_to_polygon(contour) for contour in contours]
    return filter_candidates(polygons, calibration, config)
