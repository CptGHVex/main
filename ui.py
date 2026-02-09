from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import detect
from debug_views import DebugPanel, numpy_to_qimage
from model import AcceptedRegion, AcceptedRegionsModel
from pdf_render import PdfRenderer


class CandidateGraphicsItem(QtWidgets.QGraphicsPolygonItem):
    def __init__(self, candidate: detect.CandidateRegion, parent: QtWidgets.QGraphicsItem | None = None) -> None:
        polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in candidate.polygon])
        super().__init__(polygon, parent)
        self.candidate = candidate
        self.accepted_region_id: Optional[int] = None
        self.setAcceptHoverEvents(True)
        self.update_style()

    def update_style(self) -> None:
        if self.accepted_region_id is None:
            pen = QtGui.QPen(QtGui.QColor(255, 200, 0), 2)
            brush = QtGui.QBrush(QtGui.QColor(255, 200, 0, 80))
        else:
            pen = QtGui.QPen(QtGui.QColor(0, 180, 0), 2)
            brush = QtGui.QBrush(QtGui.QColor(0, 180, 0, 100))
        self.setPen(pen)
        self.setBrush(brush)

    def set_tooltip(self, area_sqft: float, perimeter_lf: float) -> None:
        self.setToolTip(f"Area: {area_sqft:.2f} sqft\nPerimeter: {perimeter_lf:.2f} lf")


class TakeoffView(QtWidgets.QGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.main_window: Optional["MainWindow"] = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.main_window is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self.main_window.handle_scene_click(scene_pos):
                return
        super().mousePressEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, pdf_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Concrete Takeoff Prototype")
        self.resize(1400, 900)

        self.renderer: PdfRenderer | None = None
        self.current_page = 0
        self.gray_image: Optional[np.ndarray] = None
        self.rgb_image: Optional[np.ndarray] = None
        self.dpi = 600

        self.calibration: Dict[int, float] = {}
        self.calibration_points: List[QtCore.QPointF] = []
        self.calibrating = False

        self.candidates: List[CandidateGraphicsItem] = []
        self.model = AcceptedRegionsModel()
        self.last_export_dir: Optional[Path] = None

        self._build_ui()
        self._build_menu()
        self._set_empty_state()

        if pdf_path:
            self.load_pdf(pdf_path)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        self.view = TakeoffView()
        self.view.main_window = self
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view.setScene(self.scene)

        layout.addWidget(self.view, stretch=3)

        self.side_panel = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(self.side_panel)

        self.page_combo = QtWidgets.QComboBox()
        self.page_combo.currentIndexChanged.connect(self.on_page_changed)

        calibrate_button = QtWidgets.QPushButton("Calibrate")
        calibrate_button.clicked.connect(self.start_calibration)

        self.auto_detect_button = QtWidgets.QPushButton("Auto-Detect")
        self.auto_detect_button.clicked.connect(self.run_detection)

        export_button = QtWidgets.QPushButton("Export CSV")
        export_button.clicked.connect(self.export_csv)

        self.debug_checkbox = QtWidgets.QCheckBox("Debug")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug)

        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)
        self.thickness_combo = QtWidgets.QComboBox()
        self.thickness_combo.addItems(["Light", "Medium", "Heavy"])
        self.thickness_combo.setCurrentText("Medium")

        self.min_area_spin = QtWidgets.QDoubleSpinBox()
        self.min_area_spin.setRange(1.0, 10000.0)
        self.min_area_spin.setValue(20.0)
        self.min_area_spin.setSuffix(" sqft")

        self.gap_close_combo = QtWidgets.QComboBox()
        self.gap_close_combo.addItems(["Off", "Low"])
        self.gap_close_combo.setCurrentText("Low")

        settings_layout.addRow("Thickness", self.thickness_combo)
        settings_layout.addRow("Min Area", self.min_area_spin)
        settings_layout.addRow("Gap Close", self.gap_close_combo)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Page", "Region ID", "Area (sqft)", "Perimeter (lf)"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.total_area_label = QtWidgets.QLabel("Total Area: 0.00 sqft")
        self.total_perimeter_label = QtWidgets.QLabel("Total Perimeter: 0.00 lf")

        side_layout.addWidget(QtWidgets.QLabel("Page"))
        side_layout.addWidget(self.page_combo)
        side_layout.addWidget(calibrate_button)
        side_layout.addWidget(self.auto_detect_button)
        side_layout.addWidget(export_button)
        side_layout.addWidget(self.debug_checkbox)
        side_layout.addWidget(settings_group)
        side_layout.addWidget(QtWidgets.QLabel("Accepted Regions"))
        side_layout.addWidget(self.table)
        side_layout.addWidget(self.total_area_label)
        side_layout.addWidget(self.total_perimeter_label)
        side_layout.addStretch()

        layout.addWidget(self.side_panel, stretch=1)
        self.setCentralWidget(central)

        self.debug_panel = DebugPanel()
        self.debug_panel.save_requested.connect(self.save_debug_images)
        self.debug_dock = QtWidgets.QDockWidget("Debug", self)
        self.debug_dock.setWidget(self.debug_panel)
        self.debug_dock.setVisible(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.debug_dock)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        open_action = QtGui.QAction("Open PDF…", self)
        open_action.triggered.connect(self.open_pdf_dialog)
        file_menu.addAction(open_action)

        sample_action = QtGui.QAction("Open Sample PDF…", self)
        sample_action.triggered.connect(self.open_pdf_dialog)
        file_menu.addAction(sample_action)

        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _set_empty_state(self) -> None:
        self.scene.clear()
        text = self.scene.addText(
            "No PDF loaded.\n\nOpen a PDF to begin.\n\nSteps:\n1) Open PDF\n2) Select page\n3) Calibrate\n4) Auto-Detect\n5) Accept regions\n6) Export CSV"
        )
        text.setDefaultTextColor(QtGui.QColor("#666"))
        text.setPos(50, 50)

    def open_pdf_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF Files (*.pdf)"
        )
        if path:
            self.load_pdf(path)

    def load_pdf(self, path: str) -> None:
        self.renderer = PdfRenderer(path)
        self.page_combo.blockSignals(True)
        self.page_combo.clear()
        for i in range(self.renderer.page_count()):
            self.page_combo.addItem(f"Page {i + 1}", i)
        self.page_combo.blockSignals(False)
        self.current_page = 0
        self.calibration_points.clear()
        self.refresh_page()

    def refresh_page(self) -> None:
        if self.renderer is None:
            self._set_empty_state()
            return
        result = self.renderer.render_page(self.current_page, dpi=self.dpi)
        self.rgb_image = result.rgb_image
        self.gray_image = result.gray_image

        self.scene.clear()
        qimage = numpy_to_qimage(self.rgb_image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap.rect())

        self.candidates = []
        self._add_accepted_overlays()

    def _add_accepted_overlays(self) -> None:
        if self.renderer is None:
            return
        for region in self.model.regions_for_page(self.current_page):
            polygon = QtGui.QPolygonF(
                [QtCore.QPointF(x, y) for x, y in region.polygon_points]
            )
            item = QtWidgets.QGraphicsPolygonItem(polygon)
            item.setPen(QtGui.QPen(QtGui.QColor(0, 180, 0), 2))
            item.setBrush(QtGui.QBrush(QtGui.QColor(0, 180, 0, 100)))
            item.setToolTip(
                f"Area: {region.area_sqft:.2f} sqft\nPerimeter: {region.perimeter_lf:.2f} lf"
            )
            self.scene.addItem(item)

    def on_page_changed(self, index: int) -> None:
        self.current_page = index
        self.calibration_points.clear()
        self.calibrating = False
        self.refresh_page()

    def start_calibration(self) -> None:
        if self.renderer is None:
            return
        self.calibration_points = []
        self.calibrating = True
        self.statusBar().showMessage("Click two points to calibrate.")

    def handle_scene_click(self, scene_pos: QtCore.QPointF) -> bool:
        if self.renderer is None:
            return False
        if self.calibrating and len(self.calibration_points) < 2:
            self.calibration_points.append(scene_pos)
            if len(self.calibration_points) == 2:
                self.finish_calibration()
            return True
        item = self.scene.itemAt(scene_pos, QtGui.QTransform())
        if isinstance(item, CandidateGraphicsItem):
            self.toggle_candidate(item)
            return True
        return False

    def finish_calibration(self) -> None:
        if len(self.calibration_points) != 2:
            return
        dist, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibration", "Known distance (feet)", decimals=3, min=0.001
        )
        if not ok:
            self.calibration_points.clear()
            self.calibrating = False
            return
        p1, p2 = self.calibration_points
        pixel_distance = ((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2) ** 0.5
        if pixel_distance <= 0:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Invalid points selected.")
            self.calibrating = False
            return
        ft_per_pixel = dist / pixel_distance
        self.calibration[self.current_page] = ft_per_pixel
        self.statusBar().showMessage(f"Calibrated: {ft_per_pixel:.6f} ft/pixel")
        self.calibration_points.clear()
        self.calibrating = False

    def run_detection(self) -> None:
        if self.renderer is None or self.gray_image is None:
            return
        if self.current_page not in self.calibration:
            QtWidgets.QMessageBox.information(self, "Calibrate first", "Calibrate first.")
            return
        ft_per_pixel = self.calibration[self.current_page]
        candidates, debug_data = detect.detect_regions(
            gray=self.gray_image,
            thickness_preset=self.thickness_combo.currentText(),
            gap_close=self.gap_close_combo.currentText(),
            min_area_sqft=self.min_area_spin.value(),
            ft_per_pixel=ft_per_pixel,
            dpi=self.dpi,
        )
        self.clear_candidate_items()
        for candidate in candidates:
            item = CandidateGraphicsItem(candidate)
            area_sqft = candidate.area_px * (ft_per_pixel ** 2)
            perimeter_lf = candidate.perimeter_px * ft_per_pixel
            item.set_tooltip(area_sqft, perimeter_lf)
            self.scene.addItem(item)
            self.candidates.append(item)
        if self.debug_checkbox.isChecked():
            self.debug_panel.set_debug_data(debug_data)
        self.statusBar().showMessage(f"Found {len(candidates)} candidates.")

    def clear_candidate_items(self) -> None:
        for item in self.candidates:
            self.scene.removeItem(item)
        self.candidates = []

    def toggle_candidate(self, item: CandidateGraphicsItem) -> None:
        ft_per_pixel = self.calibration.get(self.current_page)
        if ft_per_pixel is None:
            QtWidgets.QMessageBox.information(self, "Calibrate first", "Calibrate first.")
            return
        if item.accepted_region_id is None:
            area_sqft = item.candidate.area_px * (ft_per_pixel ** 2)
            perimeter_lf = item.candidate.perimeter_px * ft_per_pixel
            region = self.model.add_region(
                page=self.current_page + 1,
                area_sqft=area_sqft,
                perimeter_lf=perimeter_lf,
                polygon_points=item.candidate.polygon,
            )
            item.accepted_region_id = region.region_id
            item.update_style()
        else:
            self.model.remove_region(item.accepted_region_id)
            item.accepted_region_id = None
            item.update_style()
        self.update_table()

    def update_table(self) -> None:
        regions = self.model.regions()
        self.table.setRowCount(len(regions))
        for row, region in enumerate(regions):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(region.page)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(region.region_id)))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{region.area_sqft:.2f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{region.perimeter_lf:.2f}"))
        total_area, total_perimeter = self.model.totals()
        self.total_area_label.setText(f"Total Area: {total_area:.2f} sqft")
        self.total_perimeter_label.setText(f"Total Perimeter: {total_perimeter:.2f} lf")

    def export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "takeoff.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        self.last_export_dir = Path(path).parent
        regions = self.model.regions()
        total_area, total_perimeter = self.model.totals()
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["page", "region_id", "area_sqft", "perimeter_lf"])
            for region in regions:
                writer.writerow(
                    [region.page, region.region_id, f"{region.area_sqft:.2f}", f"{region.perimeter_lf:.2f}"]
                )
            writer.writerow([])
            writer.writerow(["TOTAL", "", f"{total_area:.2f}", f"{total_perimeter:.2f}"])
        QtWidgets.QMessageBox.information(self, "Export", "CSV exported.")

    def toggle_debug(self) -> None:
        self.debug_dock.setVisible(self.debug_checkbox.isChecked())

    def save_debug_images(self) -> None:
        if not self.debug_checkbox.isChecked():
            return
        initial_dir = str(self.last_export_dir) if self.last_export_dir else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Debug Output Folder", initial_dir
        )
        if not folder:
            return
        self.debug_panel.save_images(folder)


class TakeoffApp(QtWidgets.QApplication):
    def __init__(self, pdf_path: str | None = None) -> None:
        super().__init__([])
        self.setStyle("Fusion")
        self.window = MainWindow(pdf_path=pdf_path)
        self.window.show()
