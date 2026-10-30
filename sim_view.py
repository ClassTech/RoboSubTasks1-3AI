#!/usr/bin/env python3
"""
The pygame window shared by main.py (scripted mission) and watch.py (gate
environment): a top view of the pool, a text panel, thruster bars, and the
sub's camera view with detection boxes. It draws whatever SubSim it's given;
it never changes the simulation.
"""
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from camera_geometry import distance_to, segment_polygon, to_camera, to_image, NEAR_PLANE
from config import (BLACK, CONTROL_BOX_GRAY, GRAY, GREEN, LIGHT_BLUE, ORANGE, POOL_FLOOR_COLOR, RED,
                    WATER_COLOR, YELLOW, SimulationConfig)
from core_types import ThrusterCommands

TRAIL_COLOR = (60, 60, 160)


class SimView:
    def __init__(self, config: SimulationConfig, width: int = 1200, height: int = 800,
                 title: str = "Autonomous Submarine (AUV) Simulator"):
        pygame.init()
        self.config = config
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.scale_x = width * 0.7 / config.worldWidth
        self.scale_y = height * 0.8 / config.worldHeight
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.camera_surface = pygame.Surface((config.cameraWidth, config.cameraHeight))
        self.overlay_width = max(1, config.cameraWidth // 320)  # Box line width that survives the downscaled view
        try:
            bg_img = pygame.image.load("BackgroundImage.jpg").convert()
            h = 2 * config.cameraHeight
            w = int(bg_img.get_width() * (h / bg_img.get_height()))
            self.camera_background = pygame.transform.scale(bg_img, (w, h))
            self.camera_background_pano = pygame.Surface((w * 2, h))
            self.camera_background_pano.blit(self.camera_background, (0, 0))
            self.camera_background_pano.blit(self.camera_background, (w, 0))
        except pygame.error as e:
            print(f"Error loading background image: {e}")
            self.camera_background, self.camera_background_pano = None, None

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(x * self.scale_x + 50), int((self.config.worldHeight - y) * self.scale_y + 50)

    # --- Camera view ---
    def draw_camera(self, core):
        """Draws the camera image: background, then every object at its real size, far to near."""
        s, cam, surf = core.state, core.camera, self.camera_surface
        w, h = surf.get_size()
        if self.camera_background_pano:
            bg_w, bg_h = self.camera_background.get_size()
            x_off = (s.heading / 360) * bg_w
            px_per_deg = h / 120.0  # Tuned as 2 px/deg at 240 tall
            y_off = np.clip(((bg_h - h) / 2) - (s.pitch * px_per_deg), 0, bg_h - h)
            surf.blit(self.camera_background_pano, (-x_off, -y_off))
        else:
            surf.fill(WATER_COLOR)
            floor = to_camera(s, (s.x + 20, s.y, self.config.worldDepth))
            if floor[0] > NEAR_PLANE:
                pygame.draw.rect(surf, POOL_FLOOR_COLOR, (0, to_image(cam, floor)[1], w, h))

        # Every course segment at its real size: the same geometry the detector boxes and collisions use
        drawable = []
        for sg in core.segments:
            poly = segment_polygon(s, cam, sg.p1, sg.p2, sg.radius)
            if poly:
                drawable.append((distance_to(s, sg.p1, sg.p2), sg.color, poly))
        drawable.sort(key=lambda d: d[0], reverse=True)
        limit = 4 * max(w, h)  # Keep huge coordinates of very close objects in a range pygame handles
        for _, color, poly in drawable:
            pts = [(min(max(x, -limit), limit), min(max(y, -limit), limit)) for x, y in poly]
            pygame.draw.polygon(surf, color, pts)

    def draw_boxes(self, red_boxes: Sequence[dict], gate_pair=None, green_pole: Optional[dict] = None):
        """Detection boxes on the camera view: red poles orange, the gate pair yellow, the marker green."""
        ow, surf = self.overlay_width, self.camera_surface
        for b in red_boxes:
            pygame.draw.rect(surf, ORANGE, (b['min_x'], b['min_y'], b['width'], b['height']), ow)
        if gate_pair:
            l, r = gate_pair
            min_y, max_y = min(l['min_y'], r['min_y']), max(l['max_y'], r['max_y'])
            pygame.draw.rect(surf, YELLOW, (l['min_x'], min_y, r['max_x'] - l['min_x'], max_y - min_y), ow)
        if green_pole:
            b = green_pole
            pygame.draw.rect(surf, GREEN, (b['min_x'], b['min_y'], b['width'], b['height']), 2 * ow)

    # --- Whole window ---
    def render(self, core, lines: List[str], thrusters: ThrusterCommands,
               controls: Sequence[str] = ("R - Reset", "SPACE - Pause"),
               title: str = "Autonomous Submarine Simulator (AUV)",
               trail: Optional[Sequence[Tuple[float, float]]] = None):
        screen = self.screen
        screen.fill(LIGHT_BLUE)
        pygame.draw.rect(screen, BLACK, (40, 40, int(self.config.worldWidth * self.scale_x + 20),
                                         int(self.config.worldHeight * self.scale_y + 20)), 2)
        # Top view: bars as lines, poles as dots (drawn at least 4 px so thin ones show); the
        # half-red, half-black gate poles are drawn in their top half's color
        seen = set()
        for sg in core.segments:
            a, b = sg.p1[:2], sg.p2[:2]
            if a != b:
                pygame.draw.line(screen, sg.color, self.world_to_screen(*a), self.world_to_screen(*b), 4)
                continue
            if a in seen:
                continue
            seen.add(a)
            pos, radius = self.world_to_screen(*a), max(4, int(sg.radius * self.scale_x))
            pygame.draw.circle(screen, sg.color, pos, radius)
            pygame.draw.circle(screen, BLACK, pos, radius, 1)
        if trail and len(trail) > 1:
            pygame.draw.lines(screen, TRAIL_COLOR, False, [self.world_to_screen(x, y) for x, y in trail], 2)
        self._draw_sub(core.state)

        y = 20
        screen.blit(self.font.render(title, True, BLACK), (20, y))
        y += 40
        for line in lines:
            screen.blit(self.small_font.render(line, True, BLACK), (20, y))
            y += 20
        y = self.height - 20 - 18 * (len(controls) + 1)
        for c in ("Controls:", *controls):
            screen.blit(self.small_font.render(c, True, BLACK), (20, y))
            y += 18

        tx, ty = self.width - 420, 350
        screen.blit(self.small_font.render("Thruster Output:", True, BLACK), (tx, ty))
        ty += 25
        t = thrusters
        for i, (label, v) in enumerate([("HFL", t.hfl), ("HFR", t.hfr), ("HAL", t.hal), ("HAR", t.har)]):
            self._draw_thruster_bar(tx + i * 40, ty, label, v)
        for i, (label, v) in enumerate([("VL", t.vl), ("VR", t.vr)]):
            self._draw_thruster_bar(tx + 180 + i * 40, ty, label, v)

        cam = pygame.transform.scale(self.camera_surface, (400, 300))
        if abs(core.state.roll) > 0.5:
            # The camera model ignores roll (camera_geometry), so the picture is turned as a whole: rolled
            # right side down, the scene turns counterclockwise. Scaled up so the corners stay filled.
            turned = pygame.transform.rotozoom(cam, core.state.roll, 1.67)
            cam = turned.subsurface(pygame.Rect((turned.get_width() - 400) // 2, (turned.get_height() - 300) // 2, 400, 300))
        screen.blit(cam, (self.width - 420, 20))
        pygame.draw.rect(screen, BLACK, (self.width - 420, 20, 400, 300), 2)
        self._draw_roll_gauge(core.state, self.width - 420, ty + 120)
        pygame.display.flip()

    def _draw_roll_gauge(self, s, x, y, w: int = 400, h: int = 170):
        """The sub seen from behind, rolled as it is (+ = right side down), on a level water line."""
        pygame.draw.rect(self.screen, WATER_COLOR, (x, y, w, h))
        pygame.draw.rect(self.screen, BLACK, (x, y, w, h), 2)
        cx, cy = x + w / 2, y + h / 2 + 10
        pygame.draw.line(self.screen, GRAY, (x + 10, cy), (x + w - 10, cy), 1)
        a = math.radians(s.roll)
        cos_r, sin_r = math.cos(a), math.sin(a)

        def rolled(points):   # Body (right, up) seen from behind: right side down turns it clockwise on screen
            return [(cx + dx * cos_r + dy * sin_r, cy + dx * sin_r - dy * cos_r) for dx, dy in points]

        half_w, half_h = 60, 22
        pygame.draw.polygon(self.screen, YELLOW, rolled([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]))
        pygame.draw.polygon(self.screen, BLACK, rolled([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]), 2)
        for side, label in ((-1, "L"), (1, "R")):   # The vertical thrusters, one each side
            pygame.draw.circle(self.screen, BLACK, [int(v) for v in rolled([(side * (half_w - 14), 0)])[0]], 9, 2)
            lx, ly = rolled([(side * (half_w + 14), 0)])[0]
            surf = self.small_font.render(label, True, BLACK)
            self.screen.blit(surf, (lx - surf.get_width() / 2, ly - surf.get_height() / 2))
        self.screen.blit(self.small_font.render(f"From behind: roll {s.roll:+.0f} deg, depth {s.z:.2f} m", True, BLACK),
                         (x + 8, y + 6))

    def _draw_sub(self, s):
        pos = self.world_to_screen(s.x, s.y)
        cos_h, sin_h = math.cos(math.radians(s.heading)), math.sin(math.radians(s.heading))

        def rotated(points):
            return [(pos[0] + dx * cos_h - dy * sin_h, pos[1] - (dx * sin_h + dy * cos_h)) for dx, dy in points]

        half = self.config.submarineWidth * self.scale_x / 2
        pygame.draw.polygon(self.screen, YELLOW, rotated([(-half, -half), (half, -half), (half, half), (-half, half)]), 4)
        box_w, box_l = 0.127 * self.scale_y / 2, self.config.submarineLength * self.scale_x / 2
        pygame.draw.polygon(self.screen, CONTROL_BOX_GRAY,
                            rotated([(-box_l, -box_w), (box_l, -box_w), (box_l, box_w), (-box_l, box_w)]))
        pygame.draw.polygon(self.screen, YELLOW, rotated([(box_l, -box_w), (box_l, box_w), (box_l + 0.2 * self.scale_x, 0)]))

    def _draw_thruster_bar(self, x, y, label, value):
        bar_w, bar_h = 25, 80
        center_y = y + bar_h / 2
        pygame.draw.rect(self.screen, GRAY, (x, y, bar_w, bar_h), 2)
        if value > 0:
            pygame.draw.rect(self.screen, GREEN, (x + 1, center_y - value * bar_h / 2, bar_w - 2, value * bar_h / 2))
        elif value < 0:
            pygame.draw.rect(self.screen, RED, (x + 1, center_y, bar_w - 2, abs(value) * bar_h / 2))
        pygame.draw.line(self.screen, BLACK, (x, center_y), (x + bar_w, center_y), 1)
        label_surf = self.small_font.render(label, True, BLACK)
        self.screen.blit(label_surf, (x + bar_w / 2 - label_surf.get_width() / 2, y + bar_h + 5))
