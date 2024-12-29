import os
from PIL import Image, ImageDraw
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import azure.functions as func


def detect_leaves(image_path, output_folder):
    # Load the model and processor
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    # Load and process the image
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # Perform inference
    outputs = model(**inputs)

    # Extract predictions
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 1) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label} ({round(score.item(), 2)})", fill="red")

    # Save the output image
    output_path = os.path.join(output_folder, "detected_leaves.png")
    img.save(output_path)

    return len(results["boxes"]), output_path


def main(req: func.HttpRequest) -> func.HttpResponse:
    # Input image path
    image_path = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\LeavesDetectionAlgo\test_img.jpeg'

    # Output folder
    output_folder = r'C:\Users\ChamsYAHIA\OneDrive - Arion Technologie\Documents\pi\IRM 2\sem1\iot\IOT project\iot-py-project\iot-project\LeavesDetectionLearning\LearningLeaves'
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Detect leaves
        leaf_count, output_image_path = detect_leaves(image_path, output_folder)
        return func.HttpResponse(
            f"Processed {leaf_count} leaves and saved the result to {output_image_path}.",
            status_code=200
        )
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
