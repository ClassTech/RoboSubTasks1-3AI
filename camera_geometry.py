#!/usr/bin/env python3
"""
Projects world objects into the sub's camera. Shared by the renderer (which
draws the polygons) and the simulated detector (which boxes them), so the two
always agree. Pure math, no pygame.

Objects are line segments with a radius (poles and the gate's top bar). Each
projects to a quadrilateral whose width at each end is the segment's apparent
diameter there, so near objects look thicker than far ones.
"""
import math
from typing import List, Optional, Tuple

from perception import CameraModel
from world import SubmarinePhysicsState

NEAR_PLANE = 0.2  # m; anything closer to the camera than this is not drawn

Point3 = Tuple[float, float, float]


def to_camera(state: SubmarinePhysicsState, p: Point3) -> Point3:
    """World point -> camera frame (forward, left, up). The camera sits at the sub's center."""
    dx, dy, dz = p[0] - state.x, p[1] - state.y, p[2] - state.z
    h, pt = math.radians(-state.heading), math.radians(-state.pitch)
    ch, sh, cp, sp = math.cos(h), math.sin(h), math.cos(pt), math.sin(pt)
    x_yaw, y_yaw = dx * ch - dy * sh, dx * sh + dy * ch
    return x_yaw * cp + dz * sp, y_yaw, x_yaw * sp - dz * cp


def to_image(cam: CameraModel, c: Point3) -> Tuple[float, float]:
    f = cam.focal_px
    return cam.width / 2 - f * c[1] / c[0], cam.height / 2 - f * c[2] / c[0]


def segment_polygon(state: SubmarinePhysicsState, cam: CameraModel,
                    p1: Point3, p2: Point3, radius: float) -> Optional[List[Tuple[float, float]]]:
    """Image-space quadrilateral of a segment with a radius, or None if behind the camera."""
    a, b = to_camera(state, p1), to_camera(state, p2)
    if a[0] < NEAR_PLANE and b[0] < NEAR_PLANE:
        return None
    # Clip the part behind the near plane.
    if a[0] < NEAR_PLANE or b[0] < NEAR_PLANE:
        front, back = (a, b) if a[0] >= NEAR_PLANE else (b, a)
        t = (front[0] - NEAR_PLANE) / (front[0] - back[0])
        a, b = front, tuple(front[i] + t * (back[i] - front[i]) for i in range(3))

    (ax, ay), (bx, by) = to_image(cam, a), to_image(cam, b)
    ra, rb = cam.focal_px * radius / a[0], cam.focal_px * radius / b[0]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    # Unit normal to the projected axis; a degenerate (end-on) segment gets a horizontal one.
    nx, ny = (-dy / length, dx / length) if length > 1e-9 else (1.0, 0.0)
    return [(ax + nx * ra, ay + ny * ra), (bx + nx * rb, by + ny * rb),
            (bx - nx * rb, by - ny * rb), (ax - nx * ra, ay - ny * ra)]


def distance_to(state: SubmarinePhysicsState, p1: Point3, p2: Point3) -> float:
    """Distance from the camera to a segment's midpoint, for back-to-front drawing."""
    return math.dist((state.x, state.y, state.z), tuple((p1[i] + p2[i]) / 2 for i in range(3)))
