#!/usr/bin/env python3
"""
Perception for RoboSub tasks 1-3 (the tasks course in course.py): from the
detector's red, black and white boxes to the gate's poles and passages and the
slalom's sets and lanes, as bearings, ranges and geometry. numpy and the
standard library only, so it runs on the Pi unchanged.

Gate. Each pole is half red, half black, and the two are mirrored: the right
pole (as seen from the start) is red over black, the left one black over red.
A red box and a black box stacked end to end make one pole, and their order
tells which physical pole it is, from either direction, so on the way back
the sub still knows which opening it came through. The red divider has no
black half and is never taken for a pole. The gate's top bar is never deeper
than 1.1 m and the poles are 1.52 m long, so their feet always rest on the
floor: range comes from the angle down to the foot (perception.estimate_pole's
bottom method).

Slalom. The poles float, so their feet aren't on the floor, but both ends are
usually in view and their length is known: range comes from their apparent
height. Poles at about the same range form a set; a set needs its red pole.
The lane on each side of the red pole is between it and that side's white pole.

Passages and lanes are small gates of their own: perception.gate_geometry on
their two edges (pole and divider point, or white and red pole) gives the turn
to face straight through and the sideways offset from their center line.
Sides here are relative to the sub ('left' = image left); course sides
(GatePole.course_side) are as seen from the start, facing the gate.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from perception import CameraModel, Estimate, GateGeometry, estimate_pole, gate_geometry


@dataclass
class TasksGeometry:
    """What the sub knows about the course in advance (config.TasksConfig)."""
    gate_width_m: float = 3.048
    gate_pole_diameter_m: float = 0.0334
    pool_depth_m: float = 2.09           # Where the gate poles' feet are
    slalom_pole_height_m: float = 0.9
    slalom_pole_diameter_m: float = 0.0334
    slalom_spacing_m: float = 1.524      # Red to each white
    set_range_tolerance_m: float = 1.5   # Poles within this range of each other can be one set (sets are 4 m apart)


@dataclass
class GatePole:
    box: Dict                  # Red and black boxes merged
    course_side: str           # 'right' (red over black) or 'left' (black over red), as seen from the start
    estimate: Estimate


@dataclass
class TasksGate:
    poles: List[GatePole]      # The poles found this frame, image left to right (0, 1 or 2)
    geometry: Optional[GateGeometry] = None   # The whole gate, when both poles have a range
    passages: Dict[str, GateGeometry] = field(default_factory=dict)  # 'left'/'right' relative to the sub
    divider: Optional[Dict] = None   # The divider's red box: between the poles, its top level with theirs
    used_reds: List[Dict] = field(default_factory=list)   # Red boxes that belong to the gate (poles and divider)

    @property
    def found(self) -> bool:
        return len(self.poles) == 2


@dataclass
class SlalomPole:
    box: Dict
    color: str                 # 'red' or 'white'
    estimate: Estimate


@dataclass
class SlalomSet:
    red: SlalomPole
    whites: Dict[str, SlalomPole]                            # 'left'/'right' of the red pole, relative to the sub
    lanes: Dict[str, GateGeometry] = field(default_factory=dict)

    @property
    def range_m(self) -> float:
        return self.red.estimate.range_m


# --- Gate ---
def _stacked(upper: Dict, lower: Dict, image_h: int) -> bool:
    """upper sits on top of lower, end to end, in about the same column."""
    dx = abs(upper['center_x'] - lower['center_x'])
    gap = abs(upper['max_y'] - lower['min_y'])
    return dx < (upper['width'] + lower['width']) * 1.5 + 2 and gap < 0.03 * image_h


def find_gate(reds: Sequence[Dict], blacks: Sequence[Dict], cam: CameraModel, depth_m: float,
              pitch_deg: float = 0.0, geom: Optional[TasksGeometry] = None) -> TasksGate:
    geom = geom or TasksGeometry()
    poles, used = [], set()
    for b in blacks:
        for i, r in enumerate(reds):
            if i in used:
                continue
            if _stacked(r, b, cam.height):
                side = "right"           # Red over black
            elif _stacked(b, r, cam.height):
                side = "left"            # Black over red
            else:
                continue
            used.add(i)
            box = {'min_x': min(r['min_x'], b['min_x']), 'max_x': max(r['max_x'], b['max_x']),
                   'min_y': min(r['min_y'], b['min_y']), 'max_y': max(r['max_y'], b['max_y'])}
            box.update(center_x=(box['min_x'] + box['max_x']) / 2, center_y=(box['min_y'] + box['max_y']) / 2,
                       width=box['max_x'] - box['min_x'], height=box['max_y'] - box['min_y'])
            est = estimate_pole(box, cam, diameter_m=geom.gate_pole_diameter_m,
                                bottom_depth_m=geom.pool_depth_m, camera_depth_m=depth_m, pitch_deg=pitch_deg)
            poles.append(GatePole(box, side, est))
            break
    # Keep at most one pole per course side (the taller, i.e. nearer and clearer)
    best: Dict[str, GatePole] = {}
    for p in poles:
        if p.course_side not in best or p.box['height'] > best[p.course_side].box['height']:
            best[p.course_side] = p
    poles = sorted(best.values(), key=lambda p: p.box['center_x'])
    gate = TasksGate(poles)
    gate.used_reds = [reds[i] for i in used]
    if gate.found:
        top = min(p.box['min_y'] for p in poles)
        between = [r for i, r in enumerate(reds) if i not in used
                   and poles[0].box['center_x'] < r['center_x'] < poles[1].box['center_x']
                   and abs(r['min_y'] - top) < 0.03 * cam.height]
        if between:
            gate.divider = max(between, key=lambda r: r['height'])
            gate.used_reds.append(gate.divider)
        left, right = poles[0].estimate, poles[1].estimate
        gate.geometry = gate_geometry(left, right)
        if gate.geometry is not None:
            # Each passage is a gate from its pole to the gate's center (the divider's line)
            center = _point_estimate(gate.geometry.bearing_deg, gate.geometry.range_m)
            gate.passages['left'] = gate_geometry(left, center)
            gate.passages['right'] = gate_geometry(center, right)
    return gate


def _point_estimate(bearing_deg: float, range_m: float) -> Estimate:
    return Estimate(bearing_deg, range_m, "point")


# --- Slalom ---
def find_slalom_sets(reds: Sequence[Dict], whites: Sequence[Dict], cam: CameraModel, depth_m: float,
                     pitch_deg: float = 0.0, geom: Optional[TasksGeometry] = None,
                     exclude: Sequence[Dict] = ()) -> List[SlalomSet]:
    """
    The slalom sets in view, nearest first. Red boxes in `exclude` (e.g. the
    gate's) are skipped. Only pole-shaped boxes with a range count.
    """
    geom = geom or TasksGeometry()
    excluded = {id(b) for b in exclude}
    poles = []
    for color, boxes in (("red", reds), ("white", whites)):
        for b in boxes:
            if id(b) in excluded or b['height'] <= b['width'] * 1.2:
                continue
            est = estimate_pole(b, cam, diameter_m=geom.slalom_pole_diameter_m,
                                height_m=geom.slalom_pole_height_m, pitch_deg=pitch_deg)
            if est.range_m is not None:
                poles.append(SlalomPole(b, color, est))
    sets = []
    for red in sorted((p for p in poles if p.color == "red"), key=lambda p: p.estimate.range_m):
        near = [p for p in poles if p.color == "white"
                and abs(p.estimate.range_m - red.estimate.range_m) < geom.set_range_tolerance_m]
        whites = {}
        for side, pick in (("left", lambda p: p.box['center_x'] < red.box['center_x']),
                           ("right", lambda p: p.box['center_x'] > red.box['center_x'])):
            candidates = [p for p in near if pick(p)]
            if candidates:
                whites[side] = min(candidates, key=lambda p: abs(p.box['center_x'] - red.box['center_x']))
        if not whites:
            continue   # A real set has a white pole at about the same range; a lone red is the gate's (a pole
                       # half with its black half out of view, or the divider when only one pole shows)
        s = SlalomSet(red, whites)
        for side, white in whites.items():
            lane = (gate_geometry(white.estimate, red.estimate) if side == "left"
                    else gate_geometry(red.estimate, white.estimate))
            if lane is not None:
                s.lanes[side] = lane
        sets.append(s)
    return sets
