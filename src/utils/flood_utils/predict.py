import argparse
import csv
import os
from collections import Counter
from datetime import datetime
from io import StringIO

import ee
import geemap
import pretty_errors
from dotenv import load_dotenv
from google.cloud import storage

from src.config.config import FLOOD_MODEL_ASSET_ID, TRAINING_DATA_COUNTRIES
from src.constants.constants import FLOOD_INPUT_PROPERTIES, FLOOD_SCALE
from src.utils.general_utils.data_exists import data_exists
from src.utils.general_utils.monitor_ee_tasks import monitor_tasks, start_export_task
from src.utils.general_utils.pygeoboundaries import get_adm_ee

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


training_data_countries = TRAINING_DATA_COUNTRIES
cloud_project = GOOGLE_CLOUD_PROJECT
bucket_name = GOOGLE_CLOUD_BUCKET

ee.Initialize(project=cloud_project)


def get_area_of_interest(place_name):
    """Retrieve the area of interest based on the place name."""
    return get_adm_ee(territories=place_name, adm="ADM0").geometry().bounds()


def process_data_to_classify(bbox):

    dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(bbox)
    slope = ee.Terrain.slope(dem)
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    flow_direction = ee.Image("WWF/HydroSHEDS/03DIR").clip(bbox)
    ghsl = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018").clip(bbox)

    stream_dist_proximity_collection = (
        ee.ImageCollection(
            "projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_dist_proximity"
        )
        .filterBounds(bbox)
        .mosaic()
    )
    stream_dist_proximity = stream_dist_proximity_collection.clip(bbox).rename(
        "stream_distance"
    )

    flow_accumulation_collection = (
        ee.ImageCollection(
            "projects/sat-io/open-datasets/HYDROGRAPHY90/base-network-layers/flow_accumulation"
        )
        .filterBounds(bbox)
        .mosaic()
    )
    flow_accumulation = flow_accumulation_collection.clip(bbox).rename(
        "flow_accumulation"
    )

    spi_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/spi")
        .filterBounds(bbox)
        .mosaic()
    )
    spi = spi_collection.clip(bbox).rename("spi")

    sti_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/sti")
        .filterBounds(bbox)
        .mosaic()
    )
    sti = sti_collection.clip(bbox).rename("sti")

    cti_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti")
        .filterBounds(bbox)
        .mosaic()
    )
    cti = cti_collection.clip(bbox).rename("cti")

    tpi_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/tpi")
        .filterBounds(bbox)
        .mosaic()
    )
    tpi = tpi_collection.clip(bbox).rename("tpi")

    tri_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/tri")
        .filterBounds(bbox)
        .mosaic()
    )
    tri = tri_collection.clip(bbox).rename("tri")

    pcurv_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/pcurv")
        .filterBounds(bbox)
        .mosaic()
    )
    pcurv = pcurv_collection.clip(bbox).rename("pcurv")

    tcurv_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/tcurv")
        .filterBounds(bbox)
        .mosaic()
    )
    tcurv = tcurv_collection.clip(bbox).rename("tcurv")

    aspect_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/aspect")
        .filterBounds(bbox)
        .mosaic()
    )
    aspect = aspect_collection.clip(bbox).rename("aspect")

    image_to_classify = (
        dem.toFloat()
        .rename("elevation")
        .addBands(
            landcover.select("Map").rename("landcover")
        )  # don't convert to int, so you're not exporting it
        .addBands(slope)
        .addBands(flow_direction.toFloat().rename("flow_direction"))
        .addBands(stream_dist_proximity.toFloat())
        .addBands(flow_accumulation)
        .addBands(spi.toFloat())
        .addBands(sti.toFloat())
        .addBands(cti.toFloat())
        .addBands(tpi)
        .addBands(tri)
        .addBands(pcurv)
        .addBands(tcurv)
        .addBands(aspect)
    )

    return image_to_classify


def classify_image(image_to_classify, input_properties, model_asset_id):
    """Classify the image using the pre-trained model."""
    regressor = ee.Classifier.load(model_asset_id)
    return image_to_classify.select(input_properties).classify(regressor)


def initialize_storage_client(project, bucket_name):
    """Initialize the Google Cloud Storage client."""
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    return bucket


import argparse


def export_predictions(classified_image, place_name, bucket, directory_name, scale):
    """Export the predictions to Google Cloud Storage."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    predicted_image_filename = f"predicted_flood_risk_{snake_case_place_name}"

    # Ensure the directory exists by uploading an empty file
    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    # Start export task
    task = start_export_task(
        classified_image,
        f"{place_name} predicted flood risk",
        bucket.name,
        directory_name + predicted_image_filename,
        scale,
    )
    return task


def predict(place_name):
    """Main function to predict flood risk for a given place and export the result."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    directory_name = f"data/{snake_case_place_name}/outputs/"
    predicted_image_filename = f"predicted_flood_risk_{snake_case_place_name}"

    # Check if predictions data already exists
    if data_exists(GOOGLE_CLOUD_BUCKET, directory_name + predicted_image_filename):
        print(
            f"Flood risk predictions data already exists for {place_name}. Skipping prediction."
        )
        return

    print("Processing data to classify...")
    bbox = get_area_of_interest(place_name)
    image_to_classify = process_data_to_classify(bbox)
    classified_image = classify_image(
        image_to_classify, FLOOD_INPUT_PROPERTIES, FLOOD_MODEL_ASSET_ID
    )

    # Initialize the storage client and export predictions
    bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)
    task = export_predictions(
        classified_image, place_name, bucket, directory_name, FLOOD_SCALE
    )

    # Monitor the export task
    monitor_tasks([task], 600)
