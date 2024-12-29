import os
import logging
import cv2
import numpy as np
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # Correct path to load the input image
    image_path = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\test\test_img.jpg'
    
    # Path to save the output image
    output_path = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\test\detected_leaves.jpg'
    
    # Folder to save individual leaves
    output_folder = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\test\leaves'
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if the input image exists
    if not os.path.exists(image_path):
        return func.HttpResponse("Image not found at the specified path.", status_code=404)

    # Load the image
    img = cv2.imread(image_path)

    # Resize image for faster processing (optional)
    scale_percent = 50  # Scale down to 50% of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Convert to HSV color space
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Define range for green color
    lower_green = np.array([30, 35, 35])  # Adjust as needed
    upper_green = np.array([110, 255, 255])  # Adjust as needed

    # Create a mask for green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove small noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    output = img_resized.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)  # Draw in green

    # Save the processed image with contours
    try:
        cv2.imwrite(output_path, output)
        logging.info(f"Processed image saved successfully at: {output_path}")
    except Exception as e:
        logging.error(f"Error saving the image: {e}")
        return func.HttpResponse(f"Error saving the image: {e}", status_code=500)

    # Extract and save each leaf with precise masking
    leaf_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            # Create a bounding box around each leaf
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the bounding region of interest
            roi = img_resized[y:y + h, x:x + w]

            # Create a mask for the current contour
            mask = np.zeros((h, w), dtype=np.uint8)
            contour_shifted = contour - [x, y]  # Shift contour to ROI coordinates
            cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)

            # Apply the mask to the ROI to extract the exact leaf shape
            leaf = cv2.bitwise_and(roi, roi, mask=mask)

            # Save the leaf image
            leaf_path = os.path.join(output_folder, f"leaf_{leaf_count}.png")
            try:
                cv2.imwrite(leaf_path, leaf)
                leaf_count += 1
            except Exception as e:
                logging.error(f"Error saving leaf {leaf_count}: {e}")
                return func.HttpResponse(f"Error saving leaf {leaf_count}: {e}", status_code=500)

    return func.HttpResponse(
        f"Processed {leaf_count} leaves and saved them in {output_folder}.",
        status_code=200
    )
