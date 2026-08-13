"""
MDAGT - Manual Data Augmentation via Geometric Transformations
Author: Diógenes L. Pinto
Date: 2026

This script accompanies the article:
Automatic segmentation of the ribeye area in Bos indicus and Bos taurus cattle enhanced by symmetry and homothety-based data augmentation
Journal of Animal Science, 2026

This script generates 36 augmented variations from a single image-LabelMe JSON pair
using three geometric transformations:
    1. Homothety (scaling) - 3 levels: 0.95, 1.00, 1.05
    2. Rotation - 3 angles: -5°, 0°, +5°
    3. Symmetry (flips) - 4 types: none, horizontal, vertical, both

Total: 3 × 3 × 4 = 36 augmented samples per original image.
"""

import json
import cv2
import base64
import os
import copy
import numpy as np
from pathlib import Path


# ==================== USER CONFIGURATION ====================
# Border color (BGR format) for transformed images
# Options: Green (0,255,0), White (255,255,255), Black (0,0,0)
BORDER_COLOR = (255, 255, 255)  # White - used for BTI dataset
# ============================================================


def apply_homothety_to_points(points, k, center, w, h):
    """
    Apply homothety (scaling) to a single set of polygon points.
    If k == 1.0, returns original points unchanged.

    Args:
        points: list of [x, y] coordinates from a single shape
        k: scaling factor (0.95, 1.00, or 1.05)
        center: tuple (cx, cy) - center of the image
        w: image width
        h: image height

    Returns:
        transformed_points: list of [x, y]
    """
    # Skip transformation if no scaling
    if k == 1.0:
        return points.copy()

    cx, cy = center
    new_points = []
    for x, y in points:
        new_x = k * (x - cx) + cx
        new_y = k * (y - cy) + cy
        new_points.append([new_x, new_y])
    return new_points


def apply_rotation_to_points(points, angle, center, w, h):
    """
    Apply rotation to polygon points using OpenCV's affine matrix.
    If angle == 0, returns original points unchanged.

    Rotation matrix (OpenCV format):
        M = [[cos(θ), -sin(θ), tx],
             [sin(θ),  cos(θ), ty]]

        M[0][0] = cos(θ)   - X scaling
        M[0][1] = -sin(θ)  - Y contribution to X
        M[0][2] = tx       - X translation (to keep image centered)
        M[1][0] = sin(θ)   - X contribution to Y
        M[1][1] = cos(θ)   - Y scaling
        M[1][2] = ty       - Y translation (to keep image centered)
    """
    # Skip transformation if no rotation
    if angle == 0:
        return points.copy()

    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    new_points = []
    for x, y in points:
        new_x = M[0][0] * x + M[0][1] * y + M[0][2]
        new_y = M[1][0] * x + M[1][1] * y + M[1][2]
        new_points.append([new_x, new_y])

    return new_points


def apply_symmetry_to_points(points, flip_type, w, h):
    """
    Apply symmetry (flip) to a single set of polygon points.

    Args:
        points: list of [x, y] coordinates from a single shape
        flip_type: 'none', 'h' (horizontal), 'v' (vertical), or 'hv' (both)
        w: image width
        h: image height

    Returns:
        transformed_points: list of [x, y]
    """
    if flip_type == 'none':
        return points.copy()
    elif flip_type == 'h':
        return [[w - x, y] for x, y in points]
    elif flip_type == 'v':
        return [[x, h - y] for x, y in points]
    elif flip_type == 'hv':
        return [[w - x, h - y] for x, y in points]
    else:
        raise ValueError(f"Invalid flip_type: {flip_type}")


def apply_homothety_to_image(image, k, center):
    """Apply homothety with nearest neighbor interpolation."""
    if k == 1.0:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy = center

    M = np.array([
        [k, 0, (1 - k) * cx],
        [0, k, (1 - k) * cy]
    ], dtype=np.float32)

    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderValue=BORDER_COLOR)


def apply_rotation_to_image(image, angle, center):
    """Apply rotation with nearest neighbor interpolation."""
    if angle == 0:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy = center

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderValue=BORDER_COLOR)


def apply_symmetry_to_image(image, flip_type):
    """
    Apply symmetry (flip) to image.
    If flip_type == 'none', returns original image unchanged.

    Args:
        image: numpy array (H, W, 3)
        flip_type: 'none', 'h', 'v', or 'hv'

    Returns:
        transformed_image
    """
    if flip_type == 'none':
        return image.copy()
    elif flip_type == 'h':
        return cv2.flip(image, 1)  # Horizontal flip
    elif flip_type == 'v':
        return cv2.flip(image, 0)  # Vertical flip
    elif flip_type == 'hv':
        return cv2.flip(image, -1)  # Both flips
    else:
        raise ValueError(f"Invalid flip_type: {flip_type}")


def transform_all_shapes(shapes, transform_func, *args):
    """
    Apply a transformation to all shapes in the JSON.

    Args:
        shapes: list of shape dictionaries from LabelMe JSON
        transform_func: function to apply to each shape's points
        *args: additional arguments for transform_func

    Returns:
        transformed_shapes: list of shape dictionaries with updated points
    """
    transformed_shapes = []
    for shape in shapes:
        new_shape = shape.copy()
        new_points = transform_func(shape["points"], *args)
        new_shape["points"] = new_points
        transformed_shapes.append(new_shape)
    return transformed_shapes


def clamp_points(points, w, h):
    """
    Clamp points to image boundaries after all transformations.
    Prevents out-of-bounds coordinates that would break LabelMe JSON.
    """
    clamped = []
    for x, y in points:
        clamped.append([max(0, min(w - 1, x)), max(0, min(h - 1, y))])
    return clamped


def save_image_and_json(image, json_data, output_dir, base_name, suffix):
    """
    Save transformed image and its corresponding JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_name = f"{base_name}{suffix}.jpg"
    json_name = f"{base_name}{suffix}.json"

    img_path = os.path.join(output_dir, img_name)
    json_path = os.path.join(output_dir, json_name)

    # Save image
    cv2.imwrite(img_path, image)

    # Re-encode image data for LabelMe compatibility
    with open(img_path, "rb") as f:
        json_data["imageData"] = base64.b64encode(f.read()).decode("utf-8")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)


def process_image(image_path, json_path, output_dir, k_values, angle_values, flip_types):
    """
    Main function: process a single image-JSON pair and generate all 36 variations.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        original_json = json.load(f)

    # Get image dimensions and center
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)  # Use float division for precise centering

    # Extract shapes from JSON
    original_shapes = original_json["shapes"]

    # Base name without extension
    base_name = Path(image_path).stem

    total = 0

    # Iterate through all combinations
    for k in k_values:
        for angle in angle_values:
            for flip_type in flip_types:
                # Apply transformations to image
                img_h = apply_homothety_to_image(image, k, center)
                img_r = apply_rotation_to_image(img_h, angle, center)
                img_f = apply_symmetry_to_image(img_r, flip_type)

                # Apply transformations to shapes
                shapes_h = transform_all_shapes(
                    original_shapes, apply_homothety_to_points, k, center, w, h
                )
                shapes_r = transform_all_shapes(
                    shapes_h, apply_rotation_to_points, angle, center, w, h
                )
                shapes_f = transform_all_shapes(
                    shapes_r, apply_symmetry_to_points, flip_type, w, h
                )

                # Clamp points only at the end (prevents out-of-bounds coordinates)
                shapes_clamped = []
                for shape in shapes_f:
                    new_shape = shape.copy()
                    new_shape["points"] = clamp_points(shape["points"], w, h)
                    shapes_clamped.append(new_shape)

                # Build suffix
                k_str = f"{int(k * 100):03d}"
                angle_str = f"rot{int(angle):+d}" if angle != 0 else "rot0"
                flip_map = {
                    'none': '',
                    'h': '_fliph',
                    'v': '_flipv',
                    'hv': '_fliphv'
                }
                suffix = f"_{k_str}_{angle_str}{flip_map[flip_type]}"

                # Create a fresh copy of the original JSON for each iteration
                # This prevents cumulative modifications to the same JSON object
                json_data = copy.deepcopy(original_json)

                # Update with transformed data
                json_data["imagePath"] = f"{base_name}{suffix}.jpg"
                json_data["shapes"] = shapes_clamped

                # Save image and JSON
                save_image_and_json(img_f, json_data, output_dir, base_name, suffix)

                total += 1
                print(f"Generated: {base_name}{suffix}")

    print(f"Completed! Generated {total} variations from {base_name} example")


def main():
    """
    Main execution function.

    To use this script:
    1. Set INPUT_IMAGE and INPUT_JSON paths
    2. Set OUTPUT_DIR
    3. Run the script
    """

    # ==================== USER INPUTS ====================
    # Modify these paths according to your setup

    INPUT_IMAGE = "./BTT.jpg"      # Path to input image
    INPUT_JSON = "./BTT.json"      # Path to input LabelMe JSON
    OUTPUT_DIR = "./mdagt_augmented"          # Output directory

    # Transformation parameters
    K_VALUES = [0.95, 1.00, 1.05]      # Scaling factors
    ANGLE_VALUES = [-5, 0, 5]           # Rotation angles in degrees
    FLIP_TYPES = ['none', 'h', 'v', 'hv']  # Flip types

    # ====================================================

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process the image
    process_image(INPUT_IMAGE, INPUT_JSON, OUTPUT_DIR, K_VALUES, ANGLE_VALUES, FLIP_TYPES)

    print("\nAll done! Check the output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
