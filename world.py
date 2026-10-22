#!/usr/bin/env python3
"""
Contains definitions for all physical objects in the simulation world.
"""
from dataclasses import dataclass
from typing import Tuple

# --- NEW: Pre-Qualification Course Objects ---
@dataclass
class PrequalGate:
    x: float
    center_y: float
    z_top: float      # Z-position of the top of the gate (e.g., 1.0m)
    width: float      # Width of the opening (e.g., 2.0m)
    height: float     # Height of the opening (e.g., 1.5m)
    color: Tuple[int, int, int]

@dataclass
class PrequalMarker:
    x: float
    y: float
    z_top: float      # Z-position of the top (e.g., 0.0m)
    z_bottom: float   # Z-position of the bottom (e.g., 2.1m)
    radius: float
    color: Tuple[int, int, int]


# --- Old Competition Objects (no longer used) ---
@dataclass
class Gate:
    x: float
    center_y: float
    z: float
    width: float = 3.048
    poleHeight: float = 1.524
    dividerHeight: float = 0.610
    dividerWidth: float = 0.0508
    
    @property
    def topPoleY(self) -> float: return self.center_y + self.width / 2
    
    @property
    def bottomPoleY(self) -> float: return self.center_y - self.width / 2
    
    @property
    def verticalCenterZ(self) -> float: return self.z + self.poleHeight / 2

@dataclass
class SlalomPole:
    x: float
    y: float
    z: float
    height: float = 0.9
    color: Tuple[int, int, int] = (255, 255, 255) # WHITE

@dataclass
class PathMarker:
    x: float
    y: float
    z: float
    length: float = 1.2
    width: float = 0.15
    heading: float = 0.0
    color: Tuple[int, int, int] = (255, 165, 0) # ORANGE

@dataclass
class SubmarinePhysicsState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    angular_velocity_z: float = 0.0
    angular_velocity_y: float = 0.0