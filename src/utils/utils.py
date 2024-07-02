import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO, StringIO

import ee
import geemap
import numpy as np
import pandas as pd
import pretty_errors
from dotenv import load_dotenv
from google.cloud import storage

from src.config.config import (
    EMDAT_DATA_PATH,
    FLOOD_MODEL_ASSET_ID,
    TRAINING_DATA_COUNTRIES,
)
from src.constants.constants import (
    FLOOD_INPUT_PROPERTIES,
    FLOOD_INPUTS_PATH,
    FLOOD_OUTPUTS_PATH,
    FLOOD_SCALE,
    LANDCOVER_SCALE,
)


from src.utils.pygeoboundaries.main import get_area_of_interest

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")


ee.Initialize(project=GOOGLE_CLOUD_PROJECT)


# function to initialize google cloud storage connection-------------------------------------------------------
def initialize_storage_client(project, GOOGLE_CLOUD_BUCKET):
    """Initialize the Google Cloud Storage client."""
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)
    return bucket


bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)


# function to make the place name snake case-------------------------------------------------------
def make_snake_case(place_name):
    return place_name.replace(" ", "_").lower()


# functions to start and monitor ee export tasks-------------------------------------------------------


def start_export_task(geotiff, description, bucket, fileNamePrefix, scale):
    print(f"Starting export: {description}")
    task = ee.batch.Export.image.toCloudStorage(
        image=geotiff,
        description=description,
        bucket=bucket,
        fileNamePrefix=fileNamePrefix,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def monitor_tasks(tasks, sleep_interval=10):
    """
    Monitors the completion status of provided Earth Engine tasks.

    Parameters:
    - tasks: A list of Earth Engine tasks to monitor.
    - sleep_interval: Time in seconds to wait between status checks (default is 10 seconds).
    """
    print("Monitoring tasks...")
    completed_tasks = set()
    while len(completed_tasks) < len(tasks):
        for task in tasks:
            if task.id in completed_tasks:
                continue

            try:
                status = task.status()
                state = status.get("state")

                if state in ["COMPLETED", "FAILED", "CANCELLED"]:
                    if state == "COMPLETED":
                        print(f"Task {task.id} completed successfully.")
                    elif state == "FAILED":
                        print(
                            f"Task {task.id} failed with error: {status.get('error_message', 'No error message provided.')}"
                        )
                    elif state == "CANCELLED":
                        print(f"Task {task.id} was cancelled.")
                    completed_tasks.add(task.id)
                else:
                    print(f"Task {task.id} is {state}.")
            except ee.EEException as e:
                print(f"Error checking status of task {task.id}: {e}. Will retry...")
            except Exception as general_error:
                print(f"Unexpected error: {general_error}. Will retry...")

        # Wait before the next status check to limit API requests and give time for tasks to progress
        time.sleep(sleep_interval)

    print("All tasks have been processed.")


def check_and_export_geotiffs_to_bucket(
    bucket_name, fileNamePrefix, flood_dates, bbox, scale=90
):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    existing_files = list(bucket.list_blobs(prefix=fileNamePrefix))
    existing_dates = [
        extract_date_from_filename(file.name)
        for file in existing_files
        if extract_date_from_filename(file.name) is not None
    ]

    tasks = []

    for index, (start_date, end_date) in enumerate(flood_dates):
        if start_date.strftime("%Y-%m-%d") in existing_dates:
            print(f"Skipping {start_date}: data already exist")
            continue

        training_data_result = make_training_data(bbox, start_date, end_date)
        if training_data_result is None:
            print(
                f"Skipping export for {start_date} to {end_date}: No imagery available."
            )
            continue

        geotiff = training_data_result.toShort()
        specificFileNamePrefix = f"{fileNamePrefix}input_data_{start_date}"
        export_description = f"input_data_{start_date}"

        print(
            f"Initiating export for GeoTIFF {index + 1} of {len(flood_dates)}: {export_description}"
        )
        task = start_export_task(
            geotiff, export_description, bucket_name, specificFileNamePrefix, scale
        )
        tasks.append(task)

    if tasks:
        print("All exports initiated, monitoring task status...")
        monitor_tasks(tasks)
    else:
        print("No exports were initiated.")

    print(
        f"Finished checking and exporting GeoTIFFs. Processed {len(flood_dates)} flood events."
    )


# function to check if a file or files exist before proceeding-------------------------------------------------------
def data_exists(bucket_name, prefix):
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


def list_and_check_gcs_files(bucket_name, prefix):
    """Check if files exist in a GCS bucket folder and list them if they do."""
    # Create a GCS client
    client = storage.Client()

    # Obtain the bucket object
    bucket = client.bucket(bucket_name)

    # List blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Check if any files exist with the specified prefix
    if len(blobs) == 0:
        print(f"No files found with prefix '{prefix}' in bucket '{bucket_name}'.")
        return []

    # List and return all files with the specified prefix
    file_urls = [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]
    return file_urls


def extract_date_from_filename(filename):
    # Use a regular expression to find dates in the format YYYY-MM-DD
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    else:
        return None


# function for stratified sampling based on land cover classes-------------------------------------------------------


# function to read images in a directory from GCS into an image collection-------------------------------------------------------
def convert_heat_bands_to_int(image):
    landcover_int = image.select("landcover").toInt()
    return image.addBands(landcover_int.rename("landcover"), overwrite=True)


def convert_flood_bands_to_int(image):
    """Convert the 'landcover' and 'flooded_mask' bands to integers."""
    landcover_int = image.select("landcover").toInt()
    flooded_mask_int = image.select("flooded_mask").toInt()

    return image.addBands(
        [
            landcover_int.rename("landcover"),
            flooded_mask_int.rename("flooded_mask"),
        ],
        overwrite=True,
    )


def read_images_into_collection(uri_list):
    """Read images from a list of URIs into an Earth Engine image collection."""
    ee_image_list = [ee.Image.loadGeoTIFF(url) for url in uri_list]
    image_collection = ee.ImageCollection.fromImages(ee_image_list)

    if any("flood" in uri for uri in uri_list):
        image_collection = image_collection.map(convert_flood_bands_to_int)

    if any("heat" in uri for uri in uri_list):
        image_collection = image_collection.map(convert_heat_bands_to_int)

    info = image_collection.size().getInfo()
    print(f"Collection contains {info} images.")

    return image_collection


# function to export a trained classifier-------------------------------------------------------
def export_model_as_ee_asset(regressor, description, asset_id):

    # Export the classifier
    task = ee.batch.Export.classifier.toAsset(
        classifier=regressor,
        description=description,
        assetId=asset_id,
    )
    task.start()
    print(f"Exporting trained {description} with GEE ID {asset_id}.")
    return task


# function to import a trained classifier and classify an image-------------------------------------------------------
def classify_image(image_to_classify, input_properties, model_asset_id):
    """Classify the image using the pre-trained model."""
    regressor = ee.Classifier.load(model_asset_id)
    return image_to_classify.select(input_properties).classify(regressor)


# function to make predcitions-------------------------------------------------------
def predict(
    place_name,
    predicted_image_filename,
    bucket,
    directory_name,
    scale,
    process_data_to_classify,
    input_properties,
    model_asset_id,
):
    """Main function to predict risk for a given place and export the result."""

    base_directory = f"{directory_name}{snake_case_place_name}/"

    # Check if predictions data already exists
    if data_exists(bucket.name, f"{base_directory}{predicted_image_filename}"):
        print(f"Predictions data already exists for {place_name}. Skipping prediction.")
        return

    print("Processing data to classify...")
    bbox = get_area_of_interest(place_name)
    image_to_classify = process_data_to_classify(bbox)
    classified_image = classify_image(
        image_to_classify, input_properties, model_asset_id
    )

    # Export predictions
    task = export_predictions(
        classified_image, predicted_image_filename, bucket, base_directory, scale
    )

    # Monitor the export task
    monitor_tasks([task], 600)


# def process_flood_data_to_classify(bbox):
#     """Prepare the data to be classified for flood risk."""
#     landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
#     dem = (
#         ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
#         .mosaic()
#         .clip(bbox)
#     )

#     image_to_classify = (
#         landcover.rename("landcover")
#         .addBands(dem.rename("elevation"))
#         .addBands(ee.Image.pixelLonLat())
#     )

#     return image_to_classify

# def process_heat_data_to_classify(bbox):
#     """Prepare the data to be classified for heat risk."""
#     # Add appropriate processing steps for heat data
#     pass  # Replace with actual implementation

# # Example usage for flood prediction
# predict(
#     place_name="Example Place",
#     predicted_image_filename="predicted_flood_risk_example_place",
#     bucket=initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET),
#     directory_name=FLOOD_OUTPUTS_PATH,
#     scale=FLOOD_SCALE,
#     process_data_to_classify=process_flood_data_to_classify,
#     input_properties=FLOOD_INPUT_PROPERTIES,
#     model_asset_id=FLOOD_MODEL_ASSET_ID
# )

# # Example usage for heat prediction
# predict(
#     place_name="Example Place",
#     predicted_image_filename="predicted_heat_risk_example_place",
#     bucket=initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET),
#     directory_name=HEAT_OUTPUTS_PATH,
#     scale=HEAT_SCALE,
#     process_data_to_classify=process_heat_data_to_classify,
#     input_properties=HEAT_INPUT_PROPERTIES,
#     model_asset_id=HEAT_MODEL_ASSET_ID
# )


# function to export predictions-------------------------------------------------------
def export_predictions(
    classified_image, predicted_image_filename, bucket, directory_name, scale
):
    """Export the predictions to Google Cloud Storage."""
    # Ensure the directory exists by uploading an empty file
    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    # Start export task
    task = start_export_task(
        classified_image,
        predicted_image_filename,
        bucket.name,
        directory_name + predicted_image_filename,
        scale,
    )
    return task
