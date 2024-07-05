import argparse
import csv
import json
import os
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
from src.utils.utils import (
    initialize_storage_client,
    data_exists,
    monitor_tasks,
    start_export_task,
    list_and_check_gcs_files,
    read_images_into_collection,
    classify_image,
)

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")

bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)

samples_per_flood_class = 50000


# Function to read images into collection
def read_images_into_collection(uri_list):
    ee_image_list = [ee.Image.loadGeoTIFF(url) for url in uri_list]
    image_collection = ee.ImageCollection.fromImages(ee_image_list)
    return image_collection


# Function to train and evaluate the classifier
def train_and_evaluate_classifier(image_collection, output_asset_id):
    print("Training the classifier...")

    # Sample from each image in the collection
    def sample_image(image):
        return image.sample(
            region=image.geometry(), scale=30, numPixels=5000, seed=0, geometries=True
        )

    # Aggregate samples from all images in the collection
    sampled_images = image_collection.map(sample_image)
    training_samples = sampled_images.flatten()

    classifier = ee.Classifier.smileRandomForest(10).train(
        features=training_samples,
        classProperty="flooded_mask",
        inputProperties=FLOOD_INPUT_PROPERTIES,
    )

    print("Exporting the classifier...")
    trees = ee.List(ee.Dictionary(classifier.explain()).get("trees"))
    dummy_geom = ee.Geometry.Point([0, 0])
    dummy = ee.Feature(dummy_geom)
    col = ee.FeatureCollection(trees.map(lambda x: dummy.set("tree", x)))

    task = ee.batch.Export.table.toAsset(
        collection=col,
        description="Export Flood Model",
        assetId=output_asset_id,
    )
    task.start()
    print(
        f"Export task {task.id} started, exporting trained flood model to {output_asset_id}."
    )
    monitor_tasks([task], 180)

    return output_asset_id


# Function to predict using the trained model
def predict_with_model(image, model_asset_id):
    print("Loading the classifier...")
    trees = ee.FeatureCollection(model_asset_id).aggregate_array("tree")
    classifier = ee.Classifier.decisionTreeEnsemble(trees).setOutputMode("PROBABILITY")

    print("Classifying the image...")
    classified_image = image.classify(classifier)

    return classified_image


# Main function to run the pipeline
def main(place_name):
    print(f"Generating training image URIs for {place_name}...")
    input_image_uris = []
    for country in TRAINING_DATA_COUNTRIES:
        snake_case_place_name = country.replace(" ", "_").lower()
        directory_name = f"{FLOOD_INPUTS_PATH}{snake_case_place_name}/"
        uris = list_gcs_files(GOOGLE_CLOUD_BUCKET, directory_name)
        input_image_uris.extend(uris)

    print(f"Reading images for {place_name} into collection...")
    image_collection = read_images_into_collection(input_image_uris)

    print(f"Training and evaluating classifier for {place_name}...")
    trained_model_asset_id = train_and_evaluate_classifier(
        image_collection, FLOOD_MODEL_ASSET_ID
    )

    print(f"Predicting for {place_name} using the trained model...")
    sample_image = image_collection.first()
    prediction = predict_with_model(sample_image, trained_model_asset_id)

    print(f"Prediction for {place_name} completed.")
    return prediction


def list_gcs_files(bucket_name, prefix):
    """List all files in a GCS bucket folder."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run flood prediction pipeline for a given place."
    )
    parser.add_argument(
        "place_name", type=str, help="The name of the place to predict on"
    )

    args = parser.parse_args()
    main(args.place_name)
