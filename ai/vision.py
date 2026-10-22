#!/usr/bin/env python3
"""
Contains all computer vision and image processing functions using the OpenCV library.
"""
import pygame
import cv2
import numpy as np

def find_blobs_hsv(camera_image: pygame.Surface, hsv_ranges: list, min_pixels: int):
    """
    Finds contiguous blobs of pixels matching a given set of HSV color ranges.
    This version uses the OpenCV library for high-performance image processing.
    
    Args:
        camera_image: The pygame.Surface object to process.
        hsv_ranges: A list of tuples, where each tuple contains a lower and upper HSV bound.
                    HSV values are expected in the range (H: 0-360, S: 0-100, V: 0-100).
        min_pixels: The minimum number of pixels for a blob to be considered valid.

    Returns:
        A list of dictionaries, where each dictionary describes a found blob.
    """
    # 1. Convert the Pygame Surface to a NumPy array that OpenCV can use.
    # Pygame's format is RGB, but OpenCV uses BGR, so we convert color channels.
    view = pygame.surfarray.array3d(camera_image)
    view = view.transpose([1, 0, 2])
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

    # 2. Convert the image from BGR to the HSV color space.
    # We use HSV_FULL to get a Hue range of 0-255, which is easier to scale to.
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL)

    # 3. Create a combined mask by finding all pixels that fall within any of the provided HSV ranges.
    combined_mask = None
    for lower_hsv, upper_hsv in hsv_ranges:
        # Scale our config's (360, 100, 100) format to OpenCV's (255, 255, 255) format.
        lower_bound = np.array([lower_hsv[0] * 255/360, lower_hsv[1] * 255/100, lower_hsv[2] * 255/100])
        upper_bound = np.array([upper_hsv[0] * 255/360, upper_hsv[1] * 255/100, upper_hsv[2] * 255/100])
        
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    if combined_mask is None:
        return []

    # 4. Find the contours (the outlines of the blobs) in the mask.
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blob_properties = []
    if contours:
        for cnt in contours:
            # 5. For each contour, check if its area meets the minimum size.
            area = cv2.contourArea(cnt)
            if area >= min_pixels:
                # 6. If it's big enough, get its bounding box.
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 7. Format the output to be identical to the old function's contract.
                blob_properties.append({
                    'center_x': x + w / 2,
                    'center_y': y + h / 2,
                    'height': h,
                    'width': w,
                    'min_x': x,
                    'max_x': x + w,
                    'min_y': y,
                    'max_y': y + h,
                    'area': area
                })
            
    return blob_properties