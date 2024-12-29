import os
import logging
import cv2
import numpy as np
import azure.functions as func

def save_debug_image(image, step_name, output_folder):
    """Save intermediate debug images for visualization."""
    debug_path = os.path.join(output_folder, f"debug_{step_name}.png")
    cv2.imwrite(debug_path, image)

def process_leaves(image_path: str, output_path: str, output_folder: str,
                   threshold_value: int = 0, kernel_size: int = 3,
                   dist_threshold: float = 0.7, min_region_size: int = 100) -> int:
    """Process an image to detect and segment leaves."""
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image could not be loaded from path: {image_path}")

    # Convert to HSV and create binary mask for green regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (43, 52, 52), (80, 255, 255))  # Adjust values as needed
    save_debug_image(green_mask, "binary_hsv", output_folder)

    # Apply morphological opening to clean up noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph_open = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    save_debug_image(morph_open, "morph_open_hsv", output_folder)

    # Distance transform
    dist_transform = cv2.distanceTransform(morph_open, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_debug_image(dist_norm, "distance_transform", output_folder)

    # Sure foreground
    _, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    save_debug_image(sure_fg, "sure_foreground", output_folder)

    # Sure background
    sure_bg = cv2.dilate(morph_open, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    save_debug_image(unknown, "unknown_regions", output_folder)

    # Marker labeling for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_vis = (markers.astype(np.uint8) + 1) * 50
    save_debug_image(markers_vis, "markers", output_folder)

    # Apply watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 255, 128]  # Mark watershed boundaries

    # Highlight individual regions
    for marker_id in np.unique(markers):
        if marker_id > 1:
            img[markers == marker_id] = cv2.add(img[markers == marker_id], (0, 100, 0))

    # Save final result
    cv2.imwrite(output_path, img)
    save_debug_image(img, "final_result", output_folder)

    # Extract and count leaf regions
    leaf_count = 0
    for marker_id in np.unique(markers):
        if marker_id <= 1:  # Skip background and boundary
            continue

        # Ignore small regions
        if np.count_nonzero(markers == marker_id) < min_region_size:
            continue

        # Create mask for each leaf
        mask = np.zeros_like(sure_fg, dtype=np.uint8)
        mask[markers == marker_id] = 255
        leaf = cv2.bitwise_and(img, img, mask=mask)

        # Save leaf image
        leaf_path = os.path.join(output_folder, f"leaf_{leaf_count}.png")
        cv2.imwrite(leaf_path, leaf)
        leaf_count += 1

    return leaf_count

def classify_leaves_and_flowers(output_folder: str, flower_color_lower: tuple, flower_color_upper: tuple, leaf_color_lower: tuple, leaf_color_upper: tuple):
    """Classify extracted images as leaves or flowers based on HSV color ranges."""
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_folder, file_name)
            img = cv2.imread(file_path)

            if img is None:
                continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            flower_mask = cv2.inRange(hsv, flower_color_lower, flower_color_upper)
            leaf_mask = cv2.inRange(hsv, leaf_color_lower, leaf_color_upper)

            flower_count = cv2.countNonZero(flower_mask)
            leaf_count = cv2.countNonZero(leaf_mask)

            if flower_count > leaf_count:
                new_name = file_name.replace("leaf", "flower")
                os.rename(file_path, os.path.join(output_folder, new_name))


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    image_path = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\test\test_img.jpeg'
    output_path = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\test\detected_leaves.jpg'
    output_folder = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\test\leaves'

    green_lower = (20, 25, 25)
    green_upper = (75, 225, 225)

    flower_lower = (150, 100, 100)
    flower_upper = (180, 255, 255)

    leaf_count = process_leaves(
        image_path=image_path,
        output_path=output_path,
        output_folder=output_folder,
        threshold_value=20,
        kernel_size=7,
        dist_threshold=0.8,
        min_region_size=100,
    )

    classify_leaves_and_flowers(output_folder, flower_lower, flower_upper, green_lower, green_upper)

    return func.HttpResponse(
        f"Processed {leaf_count} leaves and saved them in {output_folder}.",
        status_code=200
    )
