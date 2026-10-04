"""Output contract. `python -m propscan schema` writes schema/property_scan.schema.json.

Every measurement is a Measurement: point estimate plus a 95% interval. Intervals are
derived per tier from an explicit error budget (see propscan/uncertainty.py) and widen
as sensor data thins: LiDAR < video < photos.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class Measurement(BaseModel):
    value: float
    ci95: tuple[float, float] = Field(description="95% interval [low, high]")
    sigma: float = Field(description="1-sigma standard uncertainty, same unit")
    unit: Literal["m", "m2"] = "m"
    observed: bool = Field(True, description="False when inferred from priors (e.g. ceiling never seen)")


class Wall(BaseModel):
    id: str
    room_id: str
    surface_id: str
    start: tuple[float, float] = Field(description="plan coordinates (x, z), metres")
    end: tuple[float, float]
    length: Measurement
    height: Measurement
    area: Measurement
    shared_with: Optional[str] = Field(None, description="room on the other side, if known")


class Opening(BaseModel):
    id: str
    kind: Literal["door", "window", "passage"]
    room_ids: list[str] = Field(description="one room for exterior openings, two for connections")
    wall_ids: list[str]
    center: tuple[float, float]
    width: Measurement
    height: Optional[Measurement] = None
    sill_height: Optional[Measurement] = None
    detection_confidence: float


class Room(BaseModel):
    id: str
    name: str
    polygon: list[tuple[float, float]]
    floor_area: Measurement
    perimeter: Measurement
    ceiling_height: Measurement
    walls: list[Wall]
    opening_ids: list[str]
    floor_surface_id: str
    ceiling_surface_id: str


class DamageRegion(BaseModel):
    id: str
    surface_id: str
    damage_class: Literal["crack", "water_stain", "mold", "hole", "peeling_paint"]
    score: float
    width: Measurement
    height: Measurement
    area: Measurement
    centroid_3d: tuple[float, float, float]
    source_frames: list[int]


class ConcealedDamageFlag(BaseModel):
    id: str
    surface_id: str
    rule_id: str
    rule: str
    evidence: list[str]
    severity: Literal["low", "medium", "high"]


class ScopeLineItem(BaseModel):
    id: str
    surface_id: str
    code: str
    description: str
    quantity: Measurement
    unit: str
    reason: str


class DriftReport(BaseModel):
    method: str
    enabled: bool
    loop_closures: int
    fragments: int
    mean_correction_m: float
    max_correction_m: float
    footprint_area_uncorrected_m2: Optional[float] = None
    footprint_area_corrected_m2: Optional[float] = None


class PropertyScan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    capture_id: str
    tier: Literal["lidar", "video", "photos"]
    rooms: list[Room]
    openings: list[Opening]
    adjacency: list[tuple[str, str, str]] = Field(description="(room_a, room_b, opening_id)")
    footprint_area: Measurement
    damage: list[DamageRegion] = []
    concealed_damage: list[ConcealedDamageFlag] = []
    scope: list[ScopeLineItem] = []
    drift: Optional[DriftReport] = None
    warnings: list[str] = []
    timings_s: dict[str, float] = {}
    error_budget: dict[str, float] = Field(default_factory=dict,
                                           description="per-tier 1-sigma terms used for intervals")
