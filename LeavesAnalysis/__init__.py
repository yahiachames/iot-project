import os
import json
import datetime
import cv2  # OpenCV for image processing
from azure.data.tables import TableServiceClient, TableClient, TableEntity
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from azure.functions import HttpRequest, HttpResponse
import numpy as np

def fetch_images_from_blob():
    """
    Fetch images from Azure Blob Storage for the last 4 hours, filtered by prefix 'leaf_'.
    """
    connection_string = os.environ.get("connectionString")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(os.environ.get("containerName"))

    now = datetime.datetime.utcnow()
    four_hours_ago = now - datetime.timedelta(hours=4)

    image_paths = []
    blobs = [blob for blob in container_client.list_blobs() if "leaf_" in blob.name]
    for blob in blobs:
        try:
            # Parse the blob name into datetime components
            parts = blob.name.split('/')
            if len(parts) >= 6:
                year, month, day, hour = map(int, [parts[1], parts[3], parts[5], parts[7]])
                blob_time = datetime.datetime(year, month, day, hour)

                # Check if the blob's timestamp is within the last 4 hours
                if four_hours_ago <= blob_time <= now:
                    blob_client = container_client.get_blob_client(blob.name)
                    local_path = f"/tmp/{blob.name.replace('/', '_')}"
                    with open(local_path, "wb") as file:
                        file.write(blob_client.download_blob().readall())
                    image_paths.append(local_path)
        except Exception as e:
            print(f"Skipping blob {blob.name} due to error: {e}")

    return sorted(image_paths, key=lambda x: x.split('_')[-1])  # Sort by timestamp if embedded in filenames

def fetch_environmental_data(table_name, partition_key):
    """
    Fetch temperature and humidity data from Azure Table Storage for the last 4 hours.
    """
    connection_string = os.environ.get("connectionString")
    table_service_client = TableServiceClient.from_connection_string(connection_string)

    now = datetime.datetime.utcnow()
    four_hours_ago = now - datetime.timedelta(hours=4)
    filter_query = f"PartitionKey eq '{partition_key}' and Timestamp ge datetime'{four_hours_ago.isoformat()}'"

    temperature_data = []
    humidity_data = []

    # Use TableClient for querying entities
    table_client = table_service_client.get_table_client(table_name)
    entities = table_client.query_entities(filter_query)
    
    for entity in entities:
        if entity["Topic"] == "dht/temp":
            temperature_data.append(float(entity["Value"]))
        elif entity["Topic"] == "dht/humidity":
            humidity_data.append(float(entity["Value"]))

    return temperature_data, humidity_data

def analyze_growth(images):
    """
    Analyze plant growth from the images.
    """
    growth_rate = []
    leaf_count = []

    for img_path in images:
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the largest contour corresponds to the plant's canopy area
        canopy_area = sum(cv2.contourArea(c) for c in contours)

        # Count leaves by detecting individual contours (basic assumption)
        leaf_count.append(len(contours))
        growth_rate.append(canopy_area)

    return {
        "growth_rate": growth_rate,
        "leaf_count": leaf_count
    }

def analyze_health(images):
    """
    Analyze plant health based on leaf condition.
    """
    health_scores = []
    for img_path in images:
        image = cv2.imread(img_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect yellow/brown discoloration (stress or disease indicators)
        lower_bound = np.array([20, 100, 100])
        upper_bound = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Calculate the percentage of discolored areas
        discolored_area = cv2.countNonZero(mask)
        total_area = image.shape[0] * image.shape[1]
        health_score = 1 - (discolored_area / total_area)
        health_scores.append(health_score)

    return health_scores

def calculate_stress_thresholds(temperature_data, humidity_data):
    """
    Calculate stress thresholds for temperature and humidity.
    If no valid values are found, return 0 as the default threshold.
    """
    # For temperature, return 0 if no values satisfy the condition
    temperature_min = min([t for t in temperature_data if t < 20], default=0)
    temperature_max = max([t for t in temperature_data if t > 35], default=0)
    
    # For humidity, return 0 if no values satisfy the condition
    humidity_min = min([h for h in humidity_data if h < 30], default=0)
    humidity_max = max([h for h in humidity_data if h > 70], default=0)

    stress_thresholds = {
        "temperature": {
            "min": temperature_min,
            "max": temperature_max
        },
        "humidity": {
            "min": humidity_min,
            "max": humidity_max
        }
    }
    
    return stress_thresholds

def save_to_table_storage(table_name, entity):
    """
    Save the entity to Azure Table Storage.
    If the table doesn't exist, it will be created automatically when inserting the entity.
    """
    connection_string = os.getenv("connectionString")
    table_client = TableClient.from_connection_string(connection_string, table_name)

    try:
        # Insert the entity into the table (this will automatically create the table if it doesn't exist)
        table_client.upsert_entity(entity)
    except Exception as e:
        print(f"Error inserting entity: {str(e)}")

def main(req: HttpRequest) -> HttpResponse:
    # try:

        table_name_tempHumdity = os.environ.get("table_name_tempHumdity")

        # Validate input
        if not table_name_tempHumdity:
            return HttpResponse("Invalid table_name_tempHumdity env os", status_code=400)
        partition_key = "197001"
        # Fetch images and environmental data
        images = fetch_images_from_blob()
        temperature_data, humidity_data = fetch_environmental_data(table_name_tempHumdity, partition_key)

        # Step 1: Analyze growth and health
        growth_metrics = analyze_growth(images)
        health_scores = analyze_health(images)

        # Step 2: Calculate stress thresholds
        stress_thresholds = calculate_stress_thresholds(temperature_data, humidity_data)

        # Step 3: Save stress thresholds to Table Storage
        entity = TableEntity()
        entity["PartitionKey"] = "PlantMetrics"
        entity["RowKey"] = str(datetime.datetime.utcnow())
        entity["GrowthRate"] = json.dumps(growth_metrics["growth_rate"])
        entity["LeafCount"] = json.dumps(growth_metrics["leaf_count"])
        entity["HealthScores"] = json.dumps(health_scores)
        entity["StressThresholds"] = json.dumps(stress_thresholds)

        save_to_table_storage("PlantGrowthMetrics", entity)

        return HttpResponse(json.dumps({
            "growth_metrics": growth_metrics,
            "health_scores": health_scores,
            "stress_thresholds": stress_thresholds
        }), status_code=200, mimetype="application/json")

    # except Exception as e:
    #     return HttpResponse(f"Error: {str(e)}", status_code=500)
