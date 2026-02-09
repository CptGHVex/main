from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from detect import DetectionDebug


def numpy_to_qimage(image: np.ndarray) -> QtGui.QImage:
    if image.ndim == 2:
        height, width = image.shape
        bytes_per_line = width
        return QtGui.QImage(
            image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8
        ).copy()
    height, width, channels = image.shape
    bytes_per_line = channels * width
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(
        rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
    ).copy()


class DebugPanel(QtWidgets.QWidget):
    save_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._debug_data: DetectionDebug | None = None

        self.tabs = QtWidgets.QTabWidget()
        self._labels: Dict[str, QtWidgets.QLabel] = {}
        for title in [
            "Grayscale",
            "Binary",
            "Thickness Mask",
            "Gap Closed",
            "Contour Preview",
        ]:
            label = QtWidgets.QLabel("No data")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumSize(200, 200)
            label.setScaledContents(True)
            self._labels[title] = label
            self.tabs.addTab(label, title)

        self.stats = QtWidgets.QTextEdit()
        self.stats.setReadOnly(True)

        self.save_button = QtWidgets.QPushButton("Save Debug Images…")
        self.save_button.clicked.connect(self.save_requested.emit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(QtWidgets.QLabel("Debug Stats"))
        layout.addWidget(self.stats)
        layout.addWidget(self.save_button)

    def set_debug_data(self, debug: DetectionDebug) -> None:
        self._debug_data = debug
        images = {
            "Grayscale": debug.gray,
            "Binary": debug.binary,
            "Thickness Mask": debug.thickness_mask,
            "Gap Closed": debug.closed_mask,
            "Contour Preview": debug.contour_preview,
        }
        for title, image in images.items():
            qimage = numpy_to_qimage(image)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self._labels[title].setPixmap(pixmap)
        stats_text = "\n".join(f"{key}: {value}" for key, value in debug.stats.items())
        self.stats.setPlainText(stats_text)

    def save_images(self, folder: str | Path) -> None:
        if self._debug_data is None:
            return
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        images = {
            "01_gray.png": self._debug_data.gray,
            "02_binary.png": self._debug_data.binary,
            "03_thickness.png": self._debug_data.thickness_mask,
            "04_gap_closed.png": self._debug_data.closed_mask,
            "05_contours.png": self._debug_data.contour_preview,
        }
        for name, image in images.items():
            path = folder_path / name
            if image.ndim == 2:
                cv2.imwrite(str(path), image)
            else:
                cv2.imwrite(str(path), image)
