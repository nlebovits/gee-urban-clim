import os
from collections import Counter
from datetime import datetime

import ee
from dotenv import load_dotenv
from google.cloud import storage

from src.config.config import (
    HEAT_MODEL_ASSET_ID,
    HEAT_SCALE,
    HEAT_INPUT_PROPERTIES,
    TRAINING_DATA_COUNTRIES,
)
from src.utils.general_utils.data_exists import data_exists
from src.utils.general_utils.monitor_ee_tasks import monitor_tasks

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

training_data_countries = TRAINING_DATA_COUNTRIES
cloud_project = GOOGLE_CLOUD_PROJECT
bucket_name = GOOGLE_CLOUD_BUCKET

ee.Initialize(project=cloud_project)


def list_blobs_with_prefix(bucket_name, prefix):
    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)
    return list(bucket.list_blobs(prefix=prefix))


def read_data_to_image_collection(training_data_countries):
    bucket_name = GOOGLE_CLOUD_BUCKET
    all_tif_list = []

    # Check for data existence and collect URIs
    for country in training_data_countries:
        snake_case_place_name = country.replace(" ", "_").lower()
        directory_name = f"heat_data/{snake_case_place_name}/inputs/"

        if data_exists(bucket_name, directory_name):
            print(f"Data for {country} exists. Collecting URIs...")
            tif_list = []
            for year in range(datetime.now().year - 6, datetime.now().year - 1):
                prefix = f"{directory_name}{year}/heat_{year}"
                storage_client = storage.Client()
                blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
                for blob in blobs:
                    if blob.name.endswith(".tif"):
                        uri = f"gs://{bucket_name}/{blob.name}"
                        tif_list.append(uri)

            if tif_list:
                print(f"Collected URIs for {country}:")
                for uri in tif_list:
                    print(uri)
                all_tif_list.extend(tif_list)
            else:
                print(f"No .tif files found for {country}.")
        else:
            print(f"No data found for {country}. Skipping...")

    if all_tif_list:
        print("Reading images from cloud bucket into image collection...")
        ee_image_list = [ee.Image.loadGeoTIFF(url) for url in all_tif_list]
        image_collection = ee.ImageCollection.fromImages(ee_image_list)

        info = image_collection.size().getInfo()
        print(f"Collection contains {info} images.")
        return image_collection
    else:
        print("No data found for any country.")
        return None


def convert_landcover_to_int(image):
    landcover_int = image.select("landcover").toInt()
    return image.addBands(landcover_int.rename("landcover"), overwrite=True)


def train_and_evaluate():
    # Check if the trained model exists
    if ee.data.getInfo(HEAT_MODEL_ASSET_ID):
        print(
            f"Model already exists at {HEAT_MODEL_ASSET_ID}. Skipping training and evaluation."
        )
        return

    print("Reading data to image collection...")
    image_collection = read_data_to_image_collection(training_data_countries)

    if image_collection is None:
        print("No image collection to process.")
        return

    print("Converting landcover to integer...")
    image_collections = image_collection.map(convert_landcover_to_int)

    print("Sampling the land cover values...")
    sample = (
        image_collections.first()
        .select("landcover")
        .sample(
            scale=10,
            numPixels=10000,
            seed=0,
            geometries=False,
        )
    )

    sampled_values = sample.aggregate_array("landcover").getInfo()
    class_histogram = Counter(sampled_values)

    total_samples = 100000
    class_values = list(class_histogram.keys())
    class_points = [
        int((freq / sum(class_histogram.values())) * total_samples)
        for freq in class_histogram.values()
    ]

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

    print(
        "Initial Class Histogram:",
        {land_cover_names.get(k, k): v for k, v in class_histogram.items()},
    )

    if not class_histogram:
        print("Error: Failed to generate a class histogram.")
        raise ValueError("Failed to generate a class histogram.")

    half_total_samples = total_samples // 5
    if 50 in class_histogram:
        class_histogram[50] = half_total_samples
    if 40 in class_histogram:
        class_histogram[40] = half_total_samples
    if 30 in class_histogram:
        class_histogram[30] = half_total_samples

    print(
        "Updated Class Histogram:",
        {land_cover_names.get(k, k): v for k, v in class_histogram.items()},
    )

    class_values = list(class_histogram.keys())
    class_points = [
        int((freq / sum(class_histogram.values())) * total_samples)
        for freq in class_histogram.values()
    ]

    class_band = "landcover"
    n_images = image_collections.size().getInfo()
    samples_per_image = total_samples // n_images

    def stratified_sample_per_image(image):
        stratified_sample = image.stratifiedSample(
            numPoints=samples_per_image,
            classBand=class_band,
            scale=HEAT_SCALE,
            seed=0,
            classValues=class_values,
            classPoints=class_points,
            geometries=True,
        )
        return stratified_sample

    print("Applying stratified sampling...")
    samples = image_collections.map(stratified_sample_per_image)

    stratified_sample = samples.flatten()

    training_sample = stratified_sample.randomColumn()
    training = training_sample.filter(ee.Filter.lt("random", 0.7))
    testing = training_sample.filter(ee.Filter.gte("random", 0.7))

    inputProperties = HEAT_INPUT_PROPERTIES
    numTrees = 10
    print("Training the Random Forest regression model...")
    regressor = (
        ee.Classifier.smileRandomForest(numTrees)
        .setOutputMode("REGRESSION")
        .train(training, classProperty="median_top5", inputProperties=inputProperties)
    )

    print("Model training completed.")

    # Evaluate the classifier on the most recent image
    print("Evaluating the classifier on the most recent image...")
    sorted_filtered_collection = image_collections.sort("system:time_start", False)
    recent_image = sorted_filtered_collection.first()

    predicted_image = recent_image.select(inputProperties).classify(regressor)

    squared_difference = (
        recent_image.select("median_top5")
        .subtract(predicted_image)
        .pow(2)
        .rename("difference")
    )

    mean_squared_error = squared_difference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=recent_image.geometry(),
        scale=HEAT_SCALE,
        maxPixels=1e14,
    )

    rmse = ee.Number(mean_squared_error.get("difference")).sqrt()
    rmse_feature = ee.Feature(None, {"RMSE": rmse})

    # Step 2: Export the RMSE result
    output_path = "data/outputs/rmse_results"

    def export_results_to_cloud_storage(result, result_type, bucket_name, output_path):
        task = ee.batch.Export.table.toCloudStorage(
            collection=ee.FeatureCollection([result]),
            description=f"Export {result_type} results",
            bucket=bucket_name,
            fileNamePrefix=output_path,
            fileFormat="CSV",
        )
        task.start()
        print(
            f"Exporting {result_type} results to {output_path} in bucket {bucket_name}."
        )
        return task

    task = export_results_to_cloud_storage(
        mean_squared_error, "RMSE", bucket_name, output_path
    )

    monitor_tasks([task], 30)

    print("Exported RMSE results to cloud storage.")

    def export_model_as_ee_asset(regressor, asset_id):

        # Export the classifier
        task = ee.batch.Export.classifier.toAsset(
            classifier=regressor,
            # description="heat_rf_model_export",
            assetId=asset_id,
        )
        task.start()
        print(f"Exporting trained heat model with GEE ID {asset_id}.")
        return task

    task = export_model_as_ee_asset(regressor, HEAT_MODEL_ASSET_ID)

    monitor_tasks([task], 30)

    print(
        "Exported trained model to an Earth Engine asset with ID ", HEAT_MODEL_ASSET_ID
    )
