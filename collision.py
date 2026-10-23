#!/usr/bin/env python3
"""
Collision and gate-pass geometry. Pure math (no pygame) so it can be shared by
the graphical simulator and a headless training environment.

The sub is modeled as a vertical box: its footprint is a rectangle rotated by
heading, extended over [z - h/2, z + h/2]. Pitch is ignored. Each obstacle is a
line segment that is either vertical (a pole) or horizontal (the gate's top
bar), with a radius. For those two cases the 3D distance between the box and the
segment splits exactly into a horizontal part and a vertical part.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from course import PassLine, Segment
from world import PrequalGate


@dataclass
class Obstacle:
    name: str
    p1: Tuple[float, float]   # Horizontal (x, y) endpoints; equal for a vertical pole
    p2: Tuple[float, float]
    z_min: float              # Vertical extent of the segment
    z_max: float
    radius: float


@dataclass
class GateCrossing:
    direction: int            # +1 = crossed toward +x, -1 = toward -x
    y: float                  # Where the sub's center crossed the gate plane
    z: float
    inside_opening: bool      # Between the poles and below the top bar


@dataclass
class Pass:
    line: str                 # Pass line name (course.PassLine)
    direction: int            # +1 = crossed toward +x, -1 = toward -x
    lane: Optional[str]       # The lane crossed, or None for a miss (outside every lane)
    y: float
    z: float


def build_obstacles(segments: List[Segment]) -> List[Obstacle]:
    """Collision geometry for the course's segments (the same ones the camera draws)."""
    obstacles = []
    for sg in segments:
        if sg.collide:
            z1, z2 = sg.p1[2], sg.p2[2]
            obstacles.append(Obstacle(sg.name, sg.p1[:2], sg.p2[:2], min(z1, z2), max(z1, z2), sg.radius))
    return obstacles


def _point_rect_distance(px: float, py: float, half_l: float, half_w: float) -> float:
    """Distance from a point to an axis-aligned rectangle centered at the origin."""
    dx = max(abs(px) - half_l, 0.0)
    dy = max(abs(py) - half_w, 0.0)
    return math.hypot(dx, dy)


def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx, aby = bx - ax, by - ay
    len_sq = abx * abx + aby * aby
    t = 0.0 if len_sq == 0 else max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / len_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _segment_hits_rect(ax, ay, bx, by, half_l, half_w) -> bool:
    """Liang-Barsky clip of segment AB against the rectangle."""
    t0, t1 = 0.0, 1.0
    dx, dy = bx - ax, by - ay
    for p, q in ((-dx, ax + half_l), (dx, half_l - ax), (-dy, ay + half_w), (dy, half_w - ay)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _segment_rect_distance(a, b, half_l, half_w) -> float:
    """Distance between segment AB and the rectangle, both in the sub's frame."""
    if _segment_hits_rect(a[0], a[1], b[0], b[1], half_l, half_w):
        return 0.0
    d = min(_point_rect_distance(a[0], a[1], half_l, half_w),
            _point_rect_distance(b[0], b[1], half_l, half_w))
    for cx, cy in ((half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)):
        d = min(d, _point_segment_distance(cx, cy, a[0], a[1], b[0], b[1]))
    return d


def find_collision(x: float, y: float, z: float, heading_deg: float,
                   length: float, width: float, height: float,
                   obstacles: List[Obstacle]) -> Optional[Obstacle]:
    """Returns the first obstacle touching the sub's box, or None."""
    h = math.radians(heading_deg)
    cos_h, sin_h = math.cos(h), math.sin(h)

    def to_sub_frame(p):
        dx, dy = p[0] - x, p[1] - y
        return (dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h)

    half_l, half_w, half_h = length / 2, width / 2, height / 2
    for ob in obstacles:
        d_xy = _segment_rect_distance(to_sub_frame(ob.p1), to_sub_frame(ob.p2), half_l, half_w)
        d_z = max(ob.z_min - (z + half_h), (z - half_h) - ob.z_max, 0.0)
        if math.hypot(d_xy, d_z) < ob.radius:
            return ob
    return None


def check_gate_crossing(gate: PrequalGate, old_x: float, old_y: float, old_z: float,
                        new_x: float, new_y: float, new_z: float,
                        pole_bottom_z: float) -> Optional[GateCrossing]:
    """Detects the sub's center crossing the gate plane (x = gate.x) during one step."""
    old_side, new_side = old_x - gate.x, new_x - gate.x
    # Half-open test (x < gate.x vs x >= gate.x) so landing exactly on the plane counts once.
    if (old_side < 0) == (new_side < 0):
        return None
    t = old_side / (old_side - new_side)
    cy = old_y + t * (new_y - old_y)
    cz = old_z + t * (new_z - old_z)
    inside = abs(cy - gate.center_y) < gate.width / 2 and gate.z_top < cz < pole_bottom_z
    return GateCrossing(direction=1 if new_side > old_side else -1, y=cy, z=cz, inside_opening=inside)


def check_pass(line: PassLine, old_x: float, old_y: float, old_z: float,
               new_x: float, new_y: float, new_z: float) -> Optional[Pass]:
    """Detects the sub's center crossing a pass line during one step, and which lane it went through."""
    old_side, new_side = old_x - line.x, new_x - line.x
    if (old_side < 0) == (new_side < 0):
        return None
    t = old_side / (old_side - new_side)
    cy = old_y + t * (new_y - old_y)
    cz = old_z + t * (new_z - old_z)
    lane = None
    if line.z_min < cz < line.z_max:
        lane = next((ln.name for ln in line.lanes if ln.y_min < cy < ln.y_max), None)
    return Pass(line.name, 1 if new_side > old_side else -1, lane, cy, cz)
