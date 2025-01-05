import datetime
from azure.storage.blob import BlobServiceClient
import cv2

class BlobStorageHandler:
    def __init__(self, connection_string: str, container_name: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def save_stream_to_blob(self, stream: bytes, blob_name: str):
        """Upload a binary stream to blob storage."""
        # Encode the image as PNG
        _, buffer = cv2.imencode('.png', stream)

        # Convert to binary stream
        binary_stream = buffer.tobytes()
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(binary_stream, overwrite=True)

    def read_blob_to_stream(self, blob_name: str) -> bytes:
        """Download a blob as a binary stream."""
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()

    def get_blob_path(self, base_path: str = "") -> str:
        """Generate a blob storage path by Year/Month/Day/Hour/Minute."""
        now = datetime.datetime.now()
        path = f"year/{now.year}/month/{str(now.month).zfill(2)}/day/{str(now.day).zfill(2)}/" \
               f"hour/{str(now.hour).zfill(2)}/{base_path}"
        return path
