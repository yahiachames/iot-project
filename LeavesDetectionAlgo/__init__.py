import os
import logging
import cv2
import numpy as np
import azure.functions as func
from ..utils.BlobStorageHandler import BlobStorageHandler

def save_debug_image(image, step_name, output_folder):
    """Save intermediate debug images for visualization."""
    blob_handler = BlobStorageHandler(os.environ.get("connectionString"), os.environ.get("containerName"))
    base_path = f"{step_name}.png"
    path = blob_handler.get_blob_path(base_path)
    blob_handler.save_stream_to_blob(image,path)
 


def process_leaves(image_stream, output_folder: str,
                   threshold_value: int = 0, kernel_size: int = 3,
                   dist_threshold: float = 0.7, min_region_size: int = 100) -> int:
    """Process an image to detect and segment leaves."""
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    # Convert the byte stream into a NumPy array
    nparr = np.frombuffer(image_stream, np.uint8)

    # Decode the image from the NumPy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image from the provided stream.")


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
        save_debug_image(leaf, f"leaf_{leaf_count}", output_folder)
        leaf_count += 1

    return leaf_count


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # Get input from the HTTP request
    try:
        req_body = req.get_json()
        image_blob_name = req_body.get("imageBlobName")
        connection_string = os.environ.get("connectionString")
        container_name = os.environ.get("containerName")

        if not all([image_blob_name, connection_string, container_name]):
            return func.HttpResponse("Missing required parameters.", status_code=400)

        # Initialize Blob Storage Handler
        blob_handler = BlobStorageHandler(connection_string, container_name)

        # Download the image from Blob Storage
        image = blob_handler.read_blob_to_stream(image_blob_name)

        # Set output paths
        output_folder = "/tmp/outputs"


        # Process leaves
        leaf_count = process_leaves(image, output_folder)



        return func.HttpResponse(f"Processed {leaf_count} leaves and uploaded the outputs.", status_code=200)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)