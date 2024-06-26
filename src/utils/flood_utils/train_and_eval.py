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
    TRAINING_DATA_COUNTRIES,
    FLOOD_MODEL_ASSET_ID,
)
from src.constants.constants import LANDCOVER_SCALE, FLOOD_SCALE
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
file_name = EMDAT_DATA_PATH

ee.Initialize(project=cloud_project)

samples_per_flood_class = 50000


def aggregate_samples(
    image_collection,
    bbox,
    class_values,
    class_points,
    samples_per_image,
    flooded_status,
    landcover_band="landcover",
    flooded_mask_band="flooded_mask",
    batch_size=5,  # Number of images to process in each batch
):
    """Aggregate samples based on flooded status and export as GeoJSON."""

    def process_image(image):
        # Apply stratified sampling based on land cover classes
        condition_mask = image.select(flooded_mask_band).eq(flooded_status)
        masked_image = image.updateMask(condition_mask)

        numPoints = samples_per_image // len(class_values)
        # print(f"Number of points per class: {numPoints}")

        stratified_samples = masked_image.stratifiedSample(
            numPoints=numPoints,
            classBand=landcover_band,
            region=bbox,
            scale=FLOOD_SCALE,
            seed=0,
            geometries=True,
            classValues=class_values,
            classPoints=class_points,
        )
        return stratified_samples

    # Aggregate samples from all images in the collection in batches
    all_samples = ee.FeatureCollection([])
    image_list = image_collection.toList(image_collection.size())

    for i in range(0, image_collection.size().getInfo(), batch_size):
        batch_images = ee.ImageCollection(image_list.slice(i, i + batch_size))
        batch_samples = batch_images.map(process_image).flatten()
        all_samples = all_samples.merge(batch_samples)
        print(f"Finished processing batch {i // batch_size + 1}")

    return all_samples


def export_samples_to_gcs(samples, bucket_name, filename):
    """Export samples to Google Cloud Storage as GeoJSON."""
    try:
        task = ee.batch.Export.table.toCloudStorage(
            collection=samples,
            description="ExportToGCS",
            bucket=bucket_name,
            fileNamePrefix=filename,
            fileFormat="GeoJSON",
        )
        task.start()
        print(
            f"Export task {task.id} started, exporting samples to gs://{bucket_name}/{filename}"
        )
        return task
    except Exception as e:
        print(f"Failed to create export task: {e}")
        return None


def clean_geometry(geojson):
    """Remove unsupported 'geodesic' property from geometry definitions."""
    for feature in geojson["features"]:
        if "geodesic" in feature["geometry"]:
            del feature["geometry"]["geodesic"]
    return geojson


def read_geojson_from_gcs(bucket_name, filename):
    """Read GeoJSON file from GCS, parse, clean, and convert into ee.FeatureCollection."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    geojson_string = blob.download_as_text()

    geojson = json.loads(geojson_string)
    clean_geojson = clean_geometry(geojson)  # Clean the GeoJSON data

    geojson_fc = ee.FeatureCollection(clean_geojson)
    print("FeatureCollection created successfully.")

    return geojson_fc


def calculate_rates(confusion_matrix):
    # Convert to numpy array for easier calculations
    cm = np.array(confusion_matrix)

    TPR = []  # True Positive Rate list
    FPR = []  # False Positive Rate list

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN + cm[i, i])

        TPR.append(TP / (TP + FN) if TP + FN != 0 else 0)
        FPR.append(FP / (FP + TN) if FP + TN != 0 else 0)

    return TPR, FPR


def export_results_to_cloud_storage(
    accuracyMatrix, description, bucket_name, filePrefix
):
    """Export the error matrix to Google Cloud Storage directly."""
    # Convert the errorMatrix to a feature
    errorMatrixFeature = ee.Feature(None, {"matrix": accuracyMatrix.array()})

    # Create a FeatureCollection
    errorMatrixCollection = ee.FeatureCollection([errorMatrixFeature])

    # Start the export task
    task = ee.batch.Export.table.toCloudStorage(
        collection=errorMatrixCollection,
        description=description,
        bucket=bucket_name,
        fileNamePrefix=f"{filePrefix}",
        fileFormat="CSV",
    )
    task.start()
    print(
        f"Export task {task.id} started, exporting results to gs://{bucket_name}/{filePrefix}/{description}.csv"
    )


def train_and_evaluate_classifier(
    image_collection, bbox, bucket_name, snake_case_place_name
):
    print("Starting training and evaluation process...")
    if not isinstance(image_collection, ee.ImageCollection):
        print("Error: image_collection must be an ee.ImageCollection.")
        return None, None

    n = image_collection.size().getInfo()
    print(f"Number of images in collection: {n}")
    if n == 0:
        print("Error: Image collection is empty.")
        return None, None

    try:
        samples_per_image = int(samples_per_flood_class // n)
        print(f"Samples per flood class per image: {samples_per_image}")
        input_properties = image_collection.first().bandNames().remove("flooded_mask")
        if not input_properties:
            print("Error: No input properties after removing 'flooded_mask'.")
            return None, None

        landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
        sample = landcover.sample(
            region=bbox,
            scale=LANDCOVER_SCALE,
            numPixels=10000,
            seed=0,
            geometries=False,
        )
        sampled_values = sample.aggregate_array("Map").getInfo()

        land_cover_names = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / sparse vegetation",
            70: "Snow and ice",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
            100: "Moss and lichen",
        }

        class_histogram = Counter(sampled_values)
        print(
            "Initial Class Histogram:",
            {land_cover_names.get(k, k): v for k, v in class_histogram.items()},
        )
        if not class_histogram:
            print("Error: Failed to generate a class histogram.")
            return None, None

        # Set the "Built-up" class size equal to 1/2 the sum of all other class values
        total_class_values = sum(class_histogram.values()) / 2
        if 50 in class_histogram:  # Built-up class code
            class_histogram[50] = total_class_values
        if 40 in class_histogram:  # crop land class code
            class_histogram[40] = total_class_values
        if 30 in class_histogram:  # grassland class code
            class_histogram[30] = total_class_values

        print(
            "Updated Class Histogram:",
            {land_cover_names.get(k, k): v for k, v in class_histogram.items()},
        )

        class_values = list(class_histogram.keys())
        class_points = [
            int((freq / sum(class_histogram.values())) * samples_per_flood_class)
            for freq in class_histogram.values()
        ]

        # print class values and points for verification
        # print("Class values:", class_values)
        # print("Class points:", class_points)
        # print("Samples per flood class:", samples_per_flood_class)

        flooded_data = aggregate_samples(
            image_collection,
            bbox,
            class_values,
            class_points,
            samples_per_flood_class,
            1,
        )
        unflooded_data = aggregate_samples(
            image_collection,
            bbox,
            class_values,
            class_points,
            samples_per_flood_class,
            0,
        )
        if not flooded_data or not unflooded_data:
            print("Error: Failed to aggregate samples.")
            return None, None

        # Print sample sizes for verification
        # print("Flooded sample size:", flooded_data.size().getInfo())
        # print("Unflooded sample size:", unflooded_data.size().getInfo())

        # Add a random column to flooded and unflooded data
        flooded_data = flooded_data.randomColumn()
        unflooded_data = unflooded_data.randomColumn()

        # Sampling proportions for training, testing, and validation
        train_split = 0.6
        test_split = 0.2

        # Filter flooded data for training, testing, and validation sets
        flooded_training = flooded_data.filter(ee.Filter.lt("random", train_split))
        flooded_remaining = flooded_data.filter(ee.Filter.gte("random", train_split))

        flooded_testing = flooded_remaining.filter(
            ee.Filter.lt("random", train_split + test_split)
        )
        flooded_validation = flooded_remaining.filter(
            ee.Filter.gte("random", train_split + test_split)
        )

        # Filter unflooded data for training, testing, and validation sets
        unflooded_training = unflooded_data.filter(ee.Filter.lt("random", train_split))
        unflooded_remaining = unflooded_data.filter(
            ee.Filter.gte("random", train_split)
        )

        unflooded_testing = unflooded_remaining.filter(
            ee.Filter.lt("random", train_split + test_split)
        )
        unflooded_validation = unflooded_remaining.filter(
            ee.Filter.gte("random", train_split + test_split)
        )

        # Print sample sizes for verification
        # print("Flooded training sample size:", flooded_training.size().getInfo())
        # print("Flooded testing sample size:", flooded_testing.size().getInfo())
        # print("Flooded validation sample size:", flooded_validation.size().getInfo())

        # print("Unflooded training sample size:", unflooded_training.size().getInfo())
        # print("Unflooded testing sample size:", unflooded_testing.size().getInfo())
        # print(
        #     "Unflooded validation sample size:", unflooded_validation.size().getInfo()
        # )

        # Merge the datasets
        training_samples = flooded_training.merge(unflooded_training)
        testing_samples = flooded_testing.merge(unflooded_testing)
        validation_samples = flooded_validation.merge(unflooded_validation)

        # Print merged dataset sizes for verification
        # print("Training sample size:", training_samples.size().getInfo())
        # print("Testing sample size:", testing_samples.size().getInfo())
        # print("Validation sample size:", validation_samples.size().getInfo())

        if not training_samples or not testing_samples or not validation_samples:
            print("Error: Failed to sample datasets.")
            return None, None, None

        print("Training the classifier...")
        classifier = ee.Classifier.smileRandomForest(10).train(
            features=training_samples,
            classProperty="flooded_mask",
            inputProperties=input_properties,
        )

        print("Evaluating the classifier...")
        test_accuracy = testing_samples.classify(classifier).errorMatrix(
            "flooded_mask", "classification"
        )
        validation_accuracy = validation_samples.classify(classifier).errorMatrix(
            "flooded_mask", "classification"
        )

        print("Exporting results...")
        export_results_to_cloud_storage(
            test_accuracy,
            "Testing",
            bucket_name,
            f"data/{snake_case_place_name}/outputs/testing_results",
        )
        export_results_to_cloud_storage(
            validation_accuracy,
            "Validation",
            bucket_name,
            f"data/{snake_case_place_name}/outputs/validation_results",
        )

        print("Training and evaluation process completed.")
        return classifier, test_accuracy, validation_accuracy
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


def list_gcs_files(bucket_name, prefix):
    """List all files in a GCS bucket folder."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]


def read_images_into_collection(bucket_name, prefix):
    """Read images from cloud bucket into an Earth Engine image collection."""
    tif_list = list_gcs_files(bucket_name, prefix)
    ee_image_list = [ee.Image.loadGeoTIFF(url) for url in tif_list]
    image_collection = ee.ImageCollection.fromImages(ee_image_list)

    # Convert the 'landcover' and 'flooded_mask' bands to integers
    def convert_bands_to_int(image):
        landcover_int = image.select("landcover").toInt()
        flooded_mask_int = image.select("flooded_mask").toInt()

        return image.addBands(
            [
                landcover_int.rename("landcover"),
                flooded_mask_int.rename("flooded_mask"),
            ],
            overwrite=True,
        )

    # Apply the function to each image in the collection
    image_collection = image_collection.map(convert_bands_to_int)

    info = image_collection.size().getInfo()
    print(f"Collection contains {info} images.")

    return image_collection


def process_all_flood_data():

    if ee.data.getInfo(FLOOD_MODEL_ASSET_ID):
        print(
            f"Model already exists at {FLOOD_MODEL_ASSET_ID}. Skipping training and evaluation."
        )
        return

    combined_image_collection = ee.ImageCollection([])

    for country in training_data_countries:
        print(f"Processing flood data for {country}...")

        snake_case_place_name = country.replace(" ", "_").lower()
        directory_name = f"data/{snake_case_place_name}/inputs/"

        if not data_exists(bucket_name, directory_name):
            print(f"No training data found for {country}. Skipping...")
            continue

        print(f"Reading images for {country} into collection...")
        country_image_collection = read_images_into_collection(
            bucket_name, directory_name
        )
        if country_image_collection.size().getInfo() == 0:
            print(f"No images found for {country}. Skipping...")
            continue

        combined_image_collection = combined_image_collection.merge(
            country_image_collection
        )

    if combined_image_collection.size().getInfo() == 0:
        print("No images found in any collections. Exiting...")
        return

    print("Training and assessing model on combined data...")

    aoi = get_adm_ee(territories=training_data_countries, adm="ADM0")
    bbox = aoi.geometry().bounds()

    classifier, test_accuracy, validation_accuracy = train_and_evaluate_classifier(
        combined_image_collection, bbox, bucket_name, "combined_model"
    )
    if classifier is None:
        print("Training and evaluation failed outright. Exiting...")
        return

    def export_model_as_ee_asset(classifier, asset_id):
        task = ee.batch.Export.classifier.toAsset(
            classifier=classifier,
            assetId=asset_id,
        )
        task.start()
        print(f"Exporting trained flood model with GEE ID {asset_id}.")
        return task

    task = export_model_as_ee_asset(classifier, FLOOD_MODEL_ASSET_ID)
    monitor_tasks([task], 60)

    print("Process completed successfully.")


def main():
    process_all_flood_data()


if __name__ == "__main__":
    main()
