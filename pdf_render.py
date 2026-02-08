from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import fitz
import numpy as np


@dataclass
class RenderResult:
    rgb_image: np.ndarray
    gray_image: np.ndarray
    dpi: int


class PdfRenderer:
    def __init__(self, path: str) -> None:
        self.path = path
        self.doc = fitz.open(path)

    def page_count(self) -> int:
        return self.doc.page_count

    def render_page(self, page_index: int, dpi: int = 600) -> RenderResult:
        page = self.doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        rgb = img.copy()
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return RenderResult(rgb_image=rgb, gray_image=gray, dpi=dpi)

    def page_size(self, page_index: int) -> Tuple[float, float]:
        page = self.doc.load_page(page_index)
        rect = page.rect
        return rect.width, rect.height
