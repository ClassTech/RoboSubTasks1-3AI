#!/usr/bin/env python3
"""
The course as data: every physical object is a segment with a radius, and every
"did the sub go through it" check is a pass line. Collisions (collision.py), the
camera rendering and top view (sim_view.py) and the simulated detector
(detector_model.py) are all built from the same segment list, so they agree.

Segments are vertical (poles, the gate divider) or horizontal (the gate's top
bar). Each has a draw color and, if the camera pipeline should find it, a
detection color ('red', 'green', 'white' or 'black').

A pass line is a vertical plane x = const that the sub crosses. Its lanes are
named y ranges (and a z range): crossing inside a lane is a pass through that
lane; crossing anywhere else is a miss. Lanes are named as seen from the start,
facing +x, where +y is to the left.

Two courses:
  prequal  the qualification course: a 2 m red gate and a green marker pole
  tasks    RoboSub tasks 1-3, from ClassTech/RoboSim: a divided gate and a
           three-set slalom (see config.TasksConfig)
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import BLACK, GRAY, GREEN, RED, WHITE, PrequalConfig, SimulationConfig, TasksConfig

Point3 = Tuple[float, float, float]


@dataclass
class Segment:
    name: str
    p1: Point3
    p2: Point3
    radius: float
    color: Tuple[int, int, int]      # Draw color (RGB)
    detect: Optional[str] = None     # Detection color, or None if the detector ignores it
    collide: bool = True


@dataclass
class Lane:
    name: str
    y_min: float
    y_max: float


@dataclass
class PassLine:
    name: str
    x: float
    lanes: List[Lane]
    z_min: float                     # The lanes' vertical extent
    z_max: float


@dataclass
class Course:
    kind: str
    segments: List[Segment]
    pass_lines: List[PassLine]
    start: Dict[str, float]          # x, y, z, heading of the sub at the start
    info: Dict[str, object] = field(default_factory=dict)   # Course-specific facts (gate x, set centers, ...)

    def pass_line(self, name: str) -> PassLine:
        return next(p for p in self.pass_lines if p.name == name)


def _pole(name, x, y, z_top, z_bottom, radius, color, detect) -> Segment:
    return Segment(name, (x, y, z_top), (x, y, z_bottom), radius, color, detect)


def prequal_course(config: SimulationConfig, p: PrequalConfig, pole_bottom_z: float) -> Course:
    r = config.gatePoleRadius
    cy = config.worldHeight / 2
    half_w = p.GATE_WIDTH_METERS / 2
    gx, top = p.GATE_X_POS, p.GATE_DEPTH_METERS
    segments = [
        # Named as seen from the start side, facing +x: the +y pole is on the left
        _pole("gate right pole", gx, cy - half_w, top, pole_bottom_z, r, p.GATE_COLOR, "red"),
        _pole("gate left pole", gx, cy + half_w, top, pole_bottom_z, r, p.GATE_COLOR, "red"),
        Segment("gate top bar", (gx, cy - half_w, top), (gx, cy + half_w, top), r, GRAY),
        _pole("marker", p.MARKER_X_POS, cy, -p.POLE_ABOVE_SURFACE_METERS, pole_bottom_z,
              p.MARKER_DIAMETER_METERS / 2, p.MARKER_COLOR, "green"),
    ]
    lines = [PassLine("gate", gx, [Lane("opening", cy - half_w, cy + half_w)], top, pole_bottom_z)]
    # SubSim adds the random sideways offset (+-0.5 m) itself, at the same point in its random stream as
    # before the course was data, so the qualification run's scored test starts stay the same
    start = dict(x=p.START_X_POS, y=cy, z=p.START_Z_POS, heading=0.0)
    return Course("prequal", segments, lines, start)


def tasks_course(config: SimulationConfig, t: TasksConfig, pole_bottom_z: float,
                 rng: random.Random) -> Course:
    cy = config.worldHeight / 2
    gx = t.GATE_X_POS
    half_w = t.GATE_WIDTH_METERS / 2
    top = rng.uniform(*t.GATE_TOP_DEPTH)
    bottom = min(top + t.GATE_POLE_HEIGHT, pole_bottom_z)
    mid = (top + bottom) / 2
    r = config.gatePoleRadius
    left_y, right_y = cy + half_w, cy - half_w
    segments = [
        # Each pole is half red, half black, mirrored: red over black on the right, black over red on the left
        _pole("gate right pole top", gx, right_y, top, mid, r, RED, "red"),
        _pole("gate right pole bottom", gx, right_y, mid, bottom, r, BLACK, "black"),
        _pole("gate left pole top", gx, left_y, top, mid, r, BLACK, "black"),
        _pole("gate left pole bottom", gx, left_y, mid, bottom, r, RED, "red"),
        Segment("gate top bar", (gx, right_y, top), (gx, left_y, top), r, GRAY),
        _pole("gate divider", gx, cy, top, top + t.DIVIDER_HEIGHT, t.DIVIDER_WIDTH / 2, RED, "red"),
    ]
    lines = [PassLine("gate", gx, [Lane("right", right_y, cy), Lane("left", cy, left_y)], top, bottom)]

    s_top = rng.uniform(*t.SLALOM_TOP_DEPTH)
    s_bottom = s_top + t.SLALOM_POLE_HEIGHT
    sp, sr = t.SLALOM_POLE_SPACING, t.SLALOM_POLE_RADIUS
    margin = 2 * t.SLALOM_SET_SHIFT
    centers = []
    y = cy
    for i in range(3):
        x = gx + t.SLALOM_FIRST_X + i * t.SLALOM_SPACING_X
        if i > 0:
            y = min(max(y + rng.uniform(-t.SLALOM_SET_SHIFT, t.SLALOM_SET_SHIFT), margin),
                    config.worldHeight - margin)
        centers.append((x, y))
        segments += [
            _pole(f"slalom {i + 1} right white", x, y - sp, s_top, s_bottom, sr, WHITE, "white"),
            _pole(f"slalom {i + 1} red", x, y, s_top, s_bottom, sr, RED, "red"),
            _pole(f"slalom {i + 1} left white", x, y + sp, s_top, s_bottom, sr, WHITE, "white"),
        ]
        lines.append(PassLine(f"slalom {i + 1}", x, [Lane("right", y - sp, y), Lane("left", y, y + sp)],
                              s_top, s_bottom))

    start = dict(x=gx - t.START_BEFORE_GATE, y=cy + rng.uniform(-t.START_SIDEWAYS, t.START_SIDEWAYS),
                 z=t.START_Z_POS, heading=rng.choice(t.START_HEADINGS))
    info = dict(gate_x=gx, gate_center_y=cy, gate_top=top, gate_bottom=bottom,
                slalom_centers=centers, slalom_top=s_top, slalom_bottom=s_bottom)
    return Course("tasks", segments, lines, start, info)
