# Bid-safe Plan Takeoff Pipeline

This repository provides a reference implementation for the bid-safe pipeline described in the prompt. The focus is on keeping heavyweight outlines while filtering thin lines, then exporting geometry metrics for accepted regions.

## Pipeline Summary

1. **Render** a vector PDF page to a grayscale raster at 600 DPI.
2. **Binarize** ink vs. paper with adaptive thresholding.
3. **Thickness gate** via morphological OPEN (kernel size tied to the line-weight threshold) and optional light CLOSE.
4. **Find closed contours** (external only).
5. **Filter candidates** with bid-safe heuristics (min area, max vertex count) and a sanity score flag (perimeter/area ratio).
6. **Convert to polygons** and compute calibrated area/perimeter.
7. **Export** accepted polygons to CSV.

## Usage

```python
from pathlib import Path
from src.pipeline import Calibration, PipelineConfig, run_pipeline, export_to_csv

config = PipelineConfig(
    dpi=600,
    adaptive_block_size=35,
    adaptive_c=10,
    line_weight_threshold_px=5,
    close_kernel_px=3,
    min_area_sqft=1.0,
    max_vertex_count=1000,
    sanity_ratio_threshold=0.8,
)

calibration = Calibration(pixel_distance=512.0, real_distance_feet=10.0)
result = run_pipeline(Path("plan.pdf"), page_number=0, calibration=calibration, config=config)
export_to_csv(result.metrics, Path("output/measurements.csv"))
```

## Notes

- The pipeline does **not** auto-delete contours based on the sanity score; it only flags them.
- The thickness gate is tied to `line_weight_threshold_px`. Increase this threshold to be stricter about line weight.
