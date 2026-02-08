# PDF Concrete Takeoff Prototype (v1)

Standalone desktop prototype for concrete takeoff from PDF plans. It renders vector PDFs at high DPI, auto-detects closed regions based on line thickness, and exports accepted regions to CSV.

## Requirements
- Python 3.11+
- PySide6
- PyMuPDF (fitz)
- OpenCV
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional CLI argument to load a PDF immediately:

```bash
python main.py --pdf path/to/sample.pdf
```

## Usage Overview
1. Open a PDF.
2. Select a page.
3. Calibrate by clicking two points and entering the known distance (feet).
4. Click **Auto-Detect** to find candidate regions.
5. Click candidates to accept/unaccept.
6. Export accepted regions to CSV.

## Windows Packaging (PyInstaller)

```bash
pyinstaller --noconsole --onefile --name ConcreteTakeoff main.py
```

The executable will be in `dist/ConcreteTakeoff.exe`.

## Notes
- Rendering is done at 600 DPI.
- Measurement output is only enabled after calibration per page.
- Debug mode shows intermediate images and stats.
