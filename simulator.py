#!/usr/bin/env python3
"""
Contains the main SubmarineSimulator class: runs the scripted sub AI in the
pygame window. The physics, course, sensors and simulated detector live in
sim_core.SubSim, which runs without a display; drawing is in sim_view.SimView.
"""
import math
import time

import pygame

from ai.submarine import Submarine
from data_structures import ThrusterCommands, SensorSuite
from sim_core import SubSim
from sim_view import SimView


class SubmarineSimulator:
    def __init__(self, submarine_ai: Submarine, width=1200, height=800, detector_mode: str = "opencv"):
        """detector_mode: 'opencv' (render the camera and run OpenCV on it),
        'perfect' (simulated detector, no noise) or 'noisy' (simulated detector with noise)."""
        assert detector_mode in ("opencv", "perfect", "noisy"), detector_mode
        self.detector_mode = detector_mode
        self.core = SubSim(detector_mode=None if detector_mode == "opencv" else detector_mode)
        self.config = self.core.config
        self.view = SimView(self.config, width, height)
        self.renderCamera = True  # Headless runs with a simulated detector can skip drawing the camera image
        self.submarineAI = submarine_ai
        self.resetSimulation()

    @property
    def cameraSurface(self): return self.view.camera_surface

    def resetSimulation(self):
        self.core.reset()
        self.submarineAI.reset()
        self.startTime = time.time()
        self.running, self.paused = True, False
        self.lastThrusterCommands = ThrusterCommands()

    def handleInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_r: self.resetSimulation()

    def generateCameraView(self):
        self.view.draw_camera(self.core)

    def render(self):
        s, imu = self.core.state, self.core.true_imu
        lines = [f"Time: {time.time() - self.startTime:.1f}s",
                 f"Task: {self.submarineAI.get_current_task_name()}",
                 f"State: {self.submarineAI.get_current_state_name()}",
                 f"Speed (XY): {math.hypot(s.velocity_x, s.velocity_y):.2f} m/s",
                 f"Vel Z: {s.velocity_z:.2f} m/s",
                 f"Heading: {s.heading:.1f}°", f"Pitch: {s.pitch:.1f}°", f"Depth: {s.z:.2f} m",
                 f"Gate passes: {self.core.gate_passes}  Collisions: {self.core.collision_count}",
                 "", "IMU:",
                 f" Accel Y(surge): {imu.accel_y: .2f} m/s²",
                 f" Accel X(sway): {imu.accel_x: .2f} m/s²",
                 f" Accel Z(heave): {imu.accel_z: .2f} m/s²",
                 f" Gyro Z(yaw): {math.degrees(imu.gyro_z): .1f}°/s",
                 f" Gyro Y(pitch): {math.degrees(imu.gyro_y): .1f}°/s"]
        self.view.render(self.core, lines, self.lastThrusterCommands)

    def run(self):
        while self.running:
            dt = self.view.clock.tick(60) / 1000.0
            if dt > 0.1: dt = 0.1
            self.handleInput()
            if not self.paused:
                self.step(dt)
            self.render()
        pygame.quit()

    def step(self, dt: float):
        """ One sense / think / move cycle, without drawing the main window. """
        if self.renderCamera: self.generateCameraView()
        measured, detections = self.core.sense(dt)
        s = self.core.state
        sensors = SensorSuite(camera_image=self.cameraSurface, depth=measured.depth,
                              heading=measured.heading, pitch=measured.pitch,
                              imu=measured.imu, depth_rate=measured.depth_rate,
                              x=s.x, y=s.y, velocity_x=s.velocity_x, velocity_y=s.velocity_y,
                              angular_velocity_y=s.angular_velocity_y, velocity_z=s.velocity_z,
                              detections=detections)
        thrusterCommands, vision = self.submarineAI.update(dt, sensors)
        if thrusterCommands.pause_simulation: self.paused = True
        self.lastThrusterCommands = thrusterCommands
        self.view.draw_boxes(vision.red_blobs, vision.get_gate_pair(), vision.get_best_pole())
        self.core.advance(dt, thrusterCommands)
