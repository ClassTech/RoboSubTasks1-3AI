#!/usr/bin/env python3
"""
Defines simple data classes used for passing information between modules,
including the Vision class which holds an image provider and analyzes on demand.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable 
from detector_model import Detections
from core_types import MPU6050Readings, ThrusterCommands  # Re-exported for existing imports
from perception import find_gate_pair
import pygame
import numpy as np 
# --- Import the blob finding function ---
from ai.vision import find_blobs_hsv
# ---
# --- Import HSV Ranges (needed for thresholds potentially) ---
from config import BLACK_HSV_RANGE, GREEN_HSV_RANGE, RED_HSV_RANGES, WHITE_HSV_RANGE, SimulationConfig
# ---

@dataclass
class SensorSuite:
    # --- Measured by the real sub (with simulated noise) ---
    camera_image: pygame.Surface
    depth: float
    heading: float
    pitch: float
    imu: MPU6050Readings
    depth_rate: float = 0.0 # m/s, positive = going deeper (filtered pressure sensor)
    # Boxes from the simulated detector; None means run OpenCV on camera_image instead.
    detections: Optional[Detections] = None
    # --- Ground truth: SIMULATOR ONLY, the real sub can't measure these. ---
    # Used only by the scripted hover/damping helpers. Never feed them to the policy.
    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    angular_velocity_y: float = 0.0 # Pitch rate (rad/s)
    velocity_z: float = 0.0
    # True heading (deg), to turn the true velocity above into forward and sideways. Rotating it by
    # the measured heading instead (off by up to 15 deg per run) misread forward speed as sideways
    true_heading: Optional[float] = None

# --- UPDATED Vision Class ---
class Vision:
    """
    Handles vision processing. Gets boxes each update() (from OpenCV on the
    camera image, or from the simulated detector) and picks out the marker pole
    and the gate.

    A target that disappears is still reported, at its last seen position, for
    hold_seconds. That way a missed frame or two doesn't look like "target lost".
    A new sighting only counts once it has been seen in confirm_frames
    consecutive frames at about the same place, which rejects one-frame false
    positives.
    """
    def __init__(self, 
                 image_provider: Callable[[], pygame.Surface],
                 # Default thresholds - consider passing these from Submarine/Config
                 min_pole_pixels: int = 20, 
                 min_gate_pixels: int = 50,
                 detection_provider: Optional[Callable[[], Optional[Detections]]] = None,
                 image_width: int = 320,
                 image_height: int = 240,
                 hold_seconds: float = 0.5,
                 confirm_frames: int = 2,
                 confirm_tolerance_fraction: float = 0.05):
                 
        if not callable(image_provider):
             raise TypeError("image_provider must be a callable function")
             
        self.image_provider = image_provider
        self.detection_provider = detection_provider
        self.min_pole_pixels = min_pole_pixels
        self.min_gate_pixels = min_gate_pixels
        self.image_width = image_width
        self.image_height = image_height
        self.hold_seconds = hold_seconds
        self.confirm_frames = confirm_frames
        self.confirm_tolerance = confirm_tolerance_fraction * image_width
        self.tasks_course = False   # Set by the tasks mission: also look for white and black boxes with OpenCV
        self.reset()

    def reset(self):
        self.green_blobs: List[Dict] = []
        self.red_blobs: List[Dict] = []
        self.white_blobs: List[Dict] = []   # Tasks course: slalom poles
        self.black_blobs: List[Dict] = []   # Tasks course: the gate poles' black halves
        self._pole: Optional[Dict] = None
        self._gate_pair: Optional[Tuple[Dict, Dict]] = None
        self._pole_age = float('inf')   # Seconds since the pole was last actually seen
        self._gate_age = float('inf')
        self._pole_streak, self._last_pole_x = 0, None   # Consecutive consistent sightings
        self._gate_streak, self._last_gate_x = 0, None

    def update(self, dt: float = 0.0):
        """ Gets this frame's boxes, then updates the held pole and gate. """
        detections = self.detection_provider() if self.detection_provider else None
        if detections is not None:
             # Simulated detector: boxes are already computed, skip OpenCV.
             self.green_blobs = detections.green_blobs
             self.red_blobs = detections.red_blobs
             self.white_blobs = detections.white_blobs
             self.black_blobs = detections.black_blobs
        else:
             current_image = self.image_provider()
             if current_image is None:
                  # Handle case where provider fails (e.g., during shutdown)
                  self.green_blobs, self.red_blobs, self.white_blobs, self.black_blobs = [], [], [], []
             else:
                  self.green_blobs = find_blobs_hsv(current_image, GREEN_HSV_RANGE, self.min_pole_pixels)
                  self.red_blobs = find_blobs_hsv(current_image, RED_HSV_RANGES, self.min_gate_pixels)
                  if self.tasks_course:
                       self.white_blobs = find_blobs_hsv(current_image, WHITE_HSV_RANGE, self.min_gate_pixels)
                       self.black_blobs = find_blobs_hsv(current_image, BLACK_HSV_RANGE, self.min_gate_pixels)

        pole = self._find_best_pole()
        pole_x = pole['center_x'] if pole else None
        self._pole_streak = self._streak(self._pole_streak, self._last_pole_x, pole_x)
        self._last_pole_x = pole_x
        if pole is not None and self._pole_streak >= self.confirm_frames: self._pole, self._pole_age = pole, 0.0
        else: self._pole_age += dt

        pair = self._find_gate_pair()
        gate_x = (pair[0]['center_x'] + pair[1]['center_x']) / 2 if pair else None
        self._gate_streak = self._streak(self._gate_streak, self._last_gate_x, gate_x)
        self._last_gate_x = gate_x
        if pair is not None and self._gate_streak >= self.confirm_frames: self._gate_pair, self._gate_age = pair, 0.0
        else: self._gate_age += dt

    def _streak(self, streak: int, last_x: Optional[float], x: Optional[float]) -> int:
        if x is None: return 0
        if last_x is not None and abs(x - last_x) <= self.confirm_tolerance: return streak + 1
        return 1

    # --- Interpretation of this frame's boxes ---
    def _find_best_pole(self) -> Optional[Dict]:
        potential_poles = [b for b in self.green_blobs if b['height'] > b['width'] * 1.2] 
        return max(potential_poles, key=lambda b: b['area']) if potential_poles else None

    def _find_gate_pair(self) -> Optional[Tuple[Dict, Dict]]:
        return find_gate_pair(self.red_blobs, self.image_height)

    # --- Methods for Green Pole (Marker), with hold ---
    def get_best_pole(self) -> Optional[Dict]:
        return self._pole if self._pole_age <= self.hold_seconds else None

    def is_pole_visible(self) -> bool: return self.get_best_pole() is not None
    def get_pole_center_x(self) -> Optional[float]:
        pole = self.get_best_pole(); return pole['center_x'] if pole else None
    def get_pole_apparent_height(self) -> float:
        pole = self.get_best_pole(); return pole['height'] if pole else 0.0

    # --- Methods for Red Poles (Gate), with hold ---
    def get_gate_pair(self) -> Optional[Tuple[Dict, Dict]]:
        return self._gate_pair if self._gate_age <= self.hold_seconds else None

    def is_gate_visible(self) -> bool: return self.get_gate_pair() is not None
    def get_gate_center_x(self) -> Optional[float]:
        pair = self.get_gate_pair(); return (pair[0]['center_x'] + pair[1]['center_x']) / 2 if pair else None
