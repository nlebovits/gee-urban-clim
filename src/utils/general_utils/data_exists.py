import os

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
cloud_project = GOOGLE_CLOUD_PROJECT


# Check if data for a country exists; if it does, skip with message; if not, proceed
def data_exists(bucket_name, prefix):
    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0
