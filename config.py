#!/usr/bin/env python3
"""
Contains all constants, tuning parameters, and configuration classes for the simulation.
"""
from dataclasses import dataclass
from typing import Tuple

# --- Constants and Configuration ---
# --- MODIFIED: Added GREEN to the import ---
WHITE, BLACK, BLUE, LIGHT_BLUE = (255, 255, 255), (0, 0, 0), (20, 50, 120), (100, 150, 200)
RED, GREEN, YELLOW, GRAY = (255, 0, 0), (50, 200, 50), (255, 255, 0), (128, 128, 128)
ORANGE = (255, 165, 0)
CONTROL_BOX_GRAY = (80, 80, 80)
POOL_FLOOR_COLOR, WATER_COLOR = (40, 60, 100), (20, 50, 120)

# HSV Color Ranges for vision (H: 0-360, S: 0-100, V: 0-100)
RED_HSV_RANGES = [((0, 40, 40), (15, 100, 100)), ((340, 40, 40), (360, 100, 100))]
BLACK_HSV_RANGE = [((0, 0, 0), (360, 100, 30))]
WHITE_HSV_RANGE = [((0, 0, 70), (360, 25, 100))]
# --- NEW: HSV Range for the green marker ---
GREEN_HSV_RANGE = [((100, 40, 40), (140, 100, 100))]


@dataclass
class SimulationConfig:
    worldWidth: float = 40.0
    worldHeight: float = 15.0
    worldDepth: float = 2.1 # 7ft
    cameraFov: float = 70.0
    physicsDt: float = 1 / 60  # Fixed physics time step (s)
    cameraWidth: int = 1280
    cameraHeight: int = 960
    submarineWidth: float = 0.46
    submarineLength: float = 0.457
    submarineHeight: float = 0.25 # Collision box height (m)
    gatePoleRadius: float = 0.0167 # 1in PVC: 1.315in (33.4mm) outside diameter
    thrusterArm: float = 0.2286 # Thrusters are mounted 9in from the sub's center (yaw and pitch lever arm, m)

    # --- NEW 3D Physics Properties ---
    subMass: float = 4.0        # Mass (kg) 
    # Rotational inertia, estimated from the layout: 3/4in PVC frame (~46 cm square,
    # ~1.1 kg, plus ~0.9 kg of water if flooded), 6 thrusters (~0.7 kg) at 9in,
    # 4x9in electronics tube (~1.3 kg) at the center, 4x6x4in battery box (~0.9 kg) on top.
    subInertia_Z: float = 0.13  # Yaw (kg m^2). Mostly frame perimeter and thrusters.
    subInertia_Y: float = 0.09  # Pitch (kg m^2)
    
    thrusterMaxForce: float = 7.85 # Max force per thruster (N): 0.8 kgf, measured; the same forward and reverse.
                                   # (Was 3.56 N, read as 0.8 lbf, until 2026-10-28.)
    thrusterRampSec: float = 0.5  # The sub's software limits each thruster to 0.5 s from zero to full (and back)
    
    # Quadratic drag, force = coeff * v|v| (N per (m/s)^2), estimated as 0.5 * rho * Cd * A from the
    # open frame's parts: the 3/4in PVC tubes facing the flow, the thruster ducts, the battery box
    # and the electronics tube (Cd about 1.0-1.2 for the tubes and boxes, 0.5 for the dome). The sub is
    # blunt every way; forward is only slightly less draggy than sideways. Top speed at full thrust is
    # then about 0.5 m/s either way, and it stops almost at once. (The old values, 1.5 forward and 8
    # sideways, were from a streamlined hull: the sub coasted toward and away from the marker.)
    surgeDragCoeff: float = 60.0  # Forward/backward: about 0.12 m^2 of drag area
    swayDragCoeff: float = 70.0   # Side-to-side: the tube and battery box are longer than wide
    heaveDragCoeff: float = 80.0  # Up/down: the frame's top and bottom rectangles of tube, plus the boxes

    # Angular drag, torque = coeff * w|w| (N m per (rad/s)^2): each part's drag at its distance from
    # the center, summed (frame tubes and posts at ~0.23-0.3 m, thrusters at 9in)
    angularDragCoeff_Z: float = 1.4  # Yaw
    angularDragCoeff_Y: float = 1.4  # Pitch
    pitchRightingMoment: float = 2.0 # N m per radian of pitch: weight below buoyancy keeps it level
    # Roll (2026-10-29; nothing measured yet). The sub is statically stable in roll (user); these copy pitch's
    # estimates, the frame being about square. GUESSES: the righting moment most of all (the dance environment
    # varies it). The side vertical pair rolls it: ~3.6 N m at full and opposite thrust, against at most
    # 2.0 N m of righting (at 90 deg), so it can roll all the way over.
    subInertia_X: float = 0.09       # Roll (kg m^2)
    angularDragCoeff_X: float = 1.4  # Roll
    rollRightingMoment: float = 2.0  # N m per radian of roll
    
    # Buoyancy
    gravity: float = 9.81
    # --- MODIFIED: 0.0039 -> 0.0041 for ~1N positive buoyancy ---
    subVolume: float = 0.0041 # (m^3) e.g., 4.1L
    # ---
    waterDensity: float = 1000.0 # (kg/m^3)
    # ---

# --- Sensor Noise (all ASSUMED, TUNE ME against the real sensors) ---
# Per-episode values are drawn uniformly from (min, max) ranges at reset.
@dataclass
class SensorNoiseConfig:
    depthNoiseStd: float = 0.003            # Pressure sensor depth noise (m)
    depthOffsetRange: Tuple[float, float] = (-0.02, 0.02)   # Zero-point error (m)
    depthFilterAlpha: float = 0.3           # Alpha-beta filter gains for depth rate
    depthFilterBeta: float = 0.03           # (~0.3 s to track a step in speed)
    headingNoiseStd: float = 0.5            # Magnetometer heading noise (deg)
    headingOffsetRange: Tuple[float, float] = (-15.0, 15.0) # Pool steel distortion (deg)
    pitchNoiseStd: float = 0.2              # IMU pitch noise (deg)
    rollNoiseStd: float = 0.2               # IMU roll noise (deg); assumed as pitch
    gyroNoiseStd: float = 0.005             # Gyro noise (rad/s)
    gyroBiasRange: Tuple[float, float] = (-0.005, 0.005)    # Gyro bias (rad/s)
    accelNoiseStd: float = 0.05             # Accelerometer noise (m/s^2)

# --- Simulated Detector Noise (all ASSUMED, TUNE ME against the real camera) ---
# Ranges are (min, max), drawn uniformly per episode at reset.
@dataclass
class DetectorNoiseConfig:
    centerJitterPx: Tuple[float, float] = (0.5, 2.0)      # Box center noise (px at 320 wide; scales with resolution)
    sizeJitterFrac: Tuple[float, float] = (0.02, 0.10)    # Box width/height noise, fraction of its size
    dropoutRatePerSec: Tuple[float, float] = (0.0, 0.5)   # How often an object starts being missed
    dropoutMeanSec: float = 0.3                           # Mean length of a miss
    clearRange: Tuple[float, float] = (25.0, 35.0)        # Water clarity: detection is reliable inside this (m). Pool is very clear.
    maxRange: Tuple[float, float] = (35.0, 50.0)          # ...and impossible beyond this (m)
    falsePositivesPerSec: Tuple[float, float] = (0.0, 0.5) # Per color
    latencySec: Tuple[float, float] = (0.03, 0.15)        # Camera + processing delay

# --- Per-episode variation of the sub's dynamics (training only) ---
# Each factor is drawn uniformly from (min, max) and multiplies the nominal value, so policies
# don't come to depend on the sim's exact numbers (PID gains get retuned in the pool; inertia,
# drag and thrust are estimates).
@dataclass
class RandomizationConfig:
    pid_scale: Tuple[float, float] = (0.7, 1.3)       # Each controller gain, independently
    inertia_scale: Tuple[float, float] = (0.7, 1.3)   # Yaw and pitch inertia, independently
    drag_scale: Tuple[float, float] = (0.8, 1.2)      # Each drag coefficient, independently
    thrust_scale: Tuple[float, float] = (0.9, 1.1)    # All thrusters together

# --- Pre-Qualification Course Configuration ---
@dataclass
class PrequalConfig:
    # 1. Gate
    GATE_WIDTH_METERS: float = 2.0      # 6.6 ft
    GATE_DEPTH_METERS: float = 1.0      # 3.3 ft below surface
    GATE_OPENING_HEIGHT: float = 1.5    # Your choice, 1.5m
    GATE_COLOR: Tuple[int, int, int] = (255, 0, 0) # RED
    
    # 2. Marker
    MARKER_DIAMETER_METERS: float = 0.3 # Your choice
    MARKER_COLOR: Tuple[int, int, int] = GREEN # <-- Use constant
    
    # 3. Layout (in meters)
    GATE_X_POS: float = 10.0
    MARKER_X_POS: float = GATE_X_POS + 10.0 # 33 ft beyond
    
    # 4. Starting Position
    START_X_POS: float = 7.0 # 3m behind gate
    START_Z_POS: float = 0.1 # Start on the surface
    
    # 5. Pole Extension
    POLE_ABOVE_SURFACE_METERS: float = 0.6096
# --- Tasks 1-3 Course Configuration (from ClassTech/RoboSim, the 2025 tasks 1-2 simulator) ---
@dataclass
class TasksConfig:
    # 1. Gate: two 5 ft poles 10 ft apart, each half red and half black (mirrored, so the two sides
    #    look different), a gray top bar, and a red divider hanging from the bar's middle
    GATE_X_POS: float = 15.0
    GATE_WIDTH_METERS: float = 3.048          # 10 ft
    GATE_POLE_HEIGHT: float = 1.524           # 5 ft, down from the top bar (clipped at the floor)
    GATE_TOP_DEPTH: Tuple[float, float] = (0.6, 1.1)  # RoboSim draws 0.6-1.8 m, but below ~1.1 m the opening
                                                      # above the 2.1 m floor gets lower than ~1 m (0.3 m at 1.8)
    DIVIDER_HEIGHT: float = 0.610             # 2 ft, hanging from the top bar
    DIVIDER_WIDTH: float = 0.0508             # 2 in
    # 2. Slalom: three sets of white-red-white poles across the course
    SLALOM_FIRST_X: float = 8.0               # Past the gate
    SLALOM_SPACING_X: float = 4.0             # Between sets
    SLALOM_POLE_SPACING: float = 1.524        # 5 ft, red to each white pole
    SLALOM_SET_SHIFT: float = 1.524 * 0.25    # Each set is shifted sideways up to this from the last
    SLALOM_POLE_HEIGHT: float = 0.9
    SLALOM_TOP_DEPTH: Tuple[float, float] = (0.6, 0.9)  # Same for all sets in a run; the poles float above the floor
    SLALOM_POLE_RADIUS: float = 0.0167        # Assumed 1 in PVC, like the gate
    # 3. Start: 30 ft before the gate
    START_BEFORE_GATE: float = 9.144
    START_SIDEWAYS: float = 0.9144            # +- this from the course's center line
    START_HEADINGS: Tuple[float, ...] = (0.0, 90.0, 270.0)  # deg; 0 faces the gate
    START_Z_POS: float = 0.5
