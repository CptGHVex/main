from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AcceptedRegion:
    page: int
    region_id: int
    area_sqft: float
    perimeter_lf: float
    polygon_points: List[Tuple[float, float]] = field(default_factory=list)


class AcceptedRegionsModel:
    def __init__(self) -> None:
        self._regions: List[AcceptedRegion] = []
        self._next_id = 1

    def add_region(
        self,
        page: int,
        area_sqft: float,
        perimeter_lf: float,
        polygon_points: List[Tuple[float, float]],
    ) -> AcceptedRegion:
        region = AcceptedRegion(
            page=page,
            region_id=self._next_id,
            area_sqft=area_sqft,
            perimeter_lf=perimeter_lf,
            polygon_points=polygon_points,
        )
        self._regions.append(region)
        self._next_id += 1
        return region

    def remove_region(self, region_id: int) -> None:
        self._regions = [region for region in self._regions if region.region_id != region_id]

    def regions(self) -> List[AcceptedRegion]:
        return list(self._regions)

    def regions_for_page(self, page: int) -> List[AcceptedRegion]:
        return [region for region in self._regions if region.page == page]

    def totals(self) -> Tuple[float, float]:
        total_area = sum(region.area_sqft for region in self._regions)
        total_perimeter = sum(region.perimeter_lf for region in self._regions)
        return total_area, total_perimeter
