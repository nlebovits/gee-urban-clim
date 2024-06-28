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

from src.config.config import HEAT_MODEL_ASSET_ID, TRAINING_DATA_COUNTRIES
from src.constants.constants import (
    HEAT_INPUT_PROPERTIES,
    HEAT_INPUTS_PATH,
    HEAT_OUTPUTS_PATH,
    HEAT_SCALE,
)
from src.utils.pygeoboundaries.main import get_area_of_interest
from src.utils.utils import (
    classify_image,
    data_exists,
    export_predictions,
    initialize_storage_client,
    make_snake_case,
    monitor_tasks,
    start_export_task,
)

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")


ee.Initialize(project=GOOGLE_CLOUD_PROJECT)
bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)


def process_year(year, bbox, ndvi_min, ndvi_max):
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)
    imageCollection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(bbox)
        .filterDate(startDate, endDate)
        .map(apply_scale_factors)
        .map(cloud_mask)
    )

    def calculate_lst(image):
        ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        ndvi_min_ee = ee.Number(ndvi_min)
        ndvi_max_ee = ee.Number(ndvi_max)
        fv = (
            ee.Image()
            .expression(
                "(ndvi - ndvi_min) / (ndvi_max - ndvi_min)",
                {"ndvi": ndvi, "ndvi_max": ndvi_max_ee, "ndvi_min": ndvi_min_ee},
            )
            .pow(2)
            .rename("FV")
        )
        em = fv.multiply(ee.Number(0.004)).add(ee.Number(0.986)).rename("EM")
        thermal = image.select("ST_B10").rename("thermal")
        lst = thermal.expression(
            "(TB / (1 + (0.00115 * (TB / 1.438)) * log(em))) - 273.15",
            {
                "TB": thermal.select("thermal"),
                "em": em,
            },
        ).rename("LST")
        return lst

    lstCollection = imageCollection.map(calculate_lst)
    hotDaysCollection = lstCollection.map(lambda image: image.gte(33))
    hotDaysYear = hotDaysCollection.sum()
    hotDaysYear = hotDaysYear.rename("hot_days")
    array = lstCollection.toArray()
    sortedArray = array.arraySort().arraySlice(0, -5)
    medianOfTop5 = (
        sortedArray.arrayReduce(ee.Reducer.median(), [0])
        .arrayProject([1])
        .arrayFlatten([["median"]])
    )
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().clip(bbox)
    image_for_sampling = (
        landcover.toFloat()
        .rename("landcover")
        .addBands(dem.rename("elevation"))
        .addBands(ee.Image.pixelLonLat().toFloat())
        .addBands(medianOfTop5.toFloat().rename("median_top5"))
    )
    return image_for_sampling


def download_ndvi_data_for_year(year, snake_case_place_name):
    blob_name = f"{HEAT_INPUTS_PATH}{snake_case_place_name}/ndvi_min_max_{year}.csv"
    blob = bucket.blob(blob_name)
    ndvi_data_csv = blob.download_as_string()
    ndvi_data = csv.reader(StringIO(ndvi_data_csv.decode("utf-8")))
    rows = list(ndvi_data)
    ndvi_min = float(rows[1][1])
    ndvi_max = float(rows[1][2])
    return ndvi_min, ndvi_max


def export_ndvi_min_max(
    year, bbox, scale, snake_case_place_name, file_prefix="ndvi_min_max"
):
    file_path_prefix = f"{HEAT_INPUTS_PATH}{snake_case_place_name}/{file_prefix}_{year}"
    if data_exists(GOOGLE_CLOUD_BUCKET, file_path_prefix):
        print(f"File for {year} already exists. Skipping export.")
        return None

    try:
        startDate = ee.Date.fromYMD(year, 1, 1)
        endDate = ee.Date.fromYMD(year, 12, 31)
        imageCollection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(bbox)
            .filterDate(startDate, endDate)
            .map(apply_scale_factors)
            .map(cloud_mask)
        )
        ndviCollection = imageCollection.map(
            lambda image: image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        )
        ndvi_min = ndviCollection.reduce(ee.Reducer.min()).reduceRegion(
            reducer=ee.Reducer.min(), geometry=bbox, scale=scale, maxPixels=1e9
        )
        ndvi_max = ndviCollection.reduce(ee.Reducer.max()).reduceRegion(
            reducer=ee.Reducer.max(), geometry=bbox, scale=scale, maxPixels=1e9
        )
        feature = ee.Feature(
            None,
            {
                "ndvi_min": ndvi_min.get("NDVI_min"),
                "ndvi_max": ndvi_max.get("NDVI_max"),
            },
        )
        task = ee.batch.Export.table.toCloudStorage(
            collection=ee.FeatureCollection([feature]),
            description=f"{file_prefix}_{year}",
            bucket=GOOGLE_CLOUD_BUCKET,
            fileNamePrefix=file_path_prefix,
            fileFormat="CSV",
        )
        task.start()
        print(f"Starting export task for NDVI min/max values of year {year}.")
        return task
    except Exception as e:
        print(f"An error occurred while starting the export task for year {year}: {e}")
        return None


def apply_scale_factors(image):
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


def cloud_mask(image):
    cloud_shadow_bitmask = 1 << 3
    cloud_bitmask = 1 << 5
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(cloud_shadow_bitmask)
        .eq(0)
        .And(qa.bitwiseAnd(cloud_bitmask).eq(0))
    )
    return image.updateMask(mask)


def process_heat_data(place_name):
    current_year = datetime.now().year
    years = range(current_year - 6, current_year - 1)
    scale = HEAT_SCALE
    snake_case_place_name = make_snake_case(place_name)
    bbox = get_area_of_interest(place_name)
    directory_name = f"{HEAT_INPUTS_PATH}{snake_case_place_name}/"

    if data_exists(GOOGLE_CLOUD_BUCKET, directory_name):
        print(f"Training data for {place_name} already exists. Skipping...")
        return
    else:
        print(f"Starting to generate data for {place_name}...")

    def process_for_year(year):
        ndvi_min, ndvi_max = download_ndvi_data_for_year(year, snake_case_place_name)
        image_collection = process_year(year, bbox, ndvi_min, ndvi_max)
        return image_collection

    ndvi_tasks = []
    for year in years:
        task = export_ndvi_min_max(year, bbox, scale, snake_case_place_name)
        if task is not None:
            ndvi_tasks.append(task)

    monitor_tasks(ndvi_tasks)

    image_list = []
    tasks = []

    for year in years:
        image = process_for_year(year)
        image_list.append(image)

    for i, image in enumerate(image_list):
        year = years[i]
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=f"{snake_case_place_name}_heat_{year}",
            bucket=GOOGLE_CLOUD_BUCKET,
            fileNamePrefix=f"{directory_name}{year}/heat_{year}",
            scale=scale,
            region=bbox.getInfo()["coordinates"],
        )
        task.start()
        tasks.append(task)
        print(f"Exporting heat data for {place_name} for the year {year} to GCS...")

    monitor_tasks(tasks, 60)


def make_training_data():
    print("Generating heat risk training data from the following countries:")
    print(", ".join(TRAINING_DATA_COUNTRIES))

    for place_name in TRAINING_DATA_COUNTRIES:
        process_heat_data(place_name)

    print("Data generation completed.")


def read_data_to_image_collection(training_data_countries):
    all_tif_list = []

    for country in training_data_countries:
        snake_case_place_name = make_snake_case(country)
        directory_name = f"{HEAT_INPUTS_PATH}{snake_case_place_name}/"

        if data_exists(GOOGLE_CLOUD_BUCKET, directory_name):
            print(f"Data for {country} exists. Collecting URIs...")
            for year in range(datetime.now().year - 6, datetime.now().year - 1):
                prefix = f"{directory_name}{year}/heat_{year}"
                uris = list_and_check_gcs_files(GOOGLE_CLOUD_BUCKET, prefix)
                all_tif_list.extend(uris)

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
    if ee.data.getInfo(HEAT_MODEL_ASSET_ID):
        print(
            f"Model already exists at {HEAT_MODEL_ASSET_ID}. Skipping training and evaluation."
        )
        return

    print("Reading data to image collection...")
    image_collection = read_data_to_image_collection(TRAINING_DATA_COUNTRIES)

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

    output_path = f"{HEAT_OUTPUTS_PATH}{HEAT_MODEL_ASSET_ID}/rmse_results"

    def export_results_to_cloud_storage(result, result_type, output_path):
        task = ee.batch.Export.table.toCloudStorage(
            collection=ee.FeatureCollection([result]),
            description=f"Export {result_type} results",
            bucket=GOOGLE_CLOUD_BUCKET,
            fileNamePrefix=output_path,
            fileFormat="CSV",
        )
        task.start()
        print(
            f"Exporting {result_type} results to {output_path} in bucket {GOOGLE_CLOUD_BUCKET}."
        )
        return task

    task = export_results_to_cloud_storage(mean_squared_error, "RMSE", output_path)
    monitor_tasks([task], 30)

    print("Exported RMSE results to cloud storage.")

    def export_model_as_ee_asset(regressor, asset_id):
        task = ee.batch.Export.classifier.toAsset(
            classifier=regressor,
            assetId=asset_id,
        )
        task.start()
        print(f"Exporting trained heat model with GEE ID {asset_id}.")
        return task

    task = export_model_as_ee_asset(regressor, HEAT_MODEL_ASSET_ID)
    monitor_tasks([task], 30)

    print(
        f"Exported trained model to an Earth Engine asset with ID {HEAT_MODEL_ASSET_ID}."
    )


def process_data_to_classify(bbox):
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().clip(bbox)

    image_to_classify = (
        landcover.rename("landcover")
        .addBands(dem.rename("elevation"))
        .addBands(ee.Image.pixelLonLat())
    )

    return image_to_classify


def predict(place_name):
    snake_case_place_name = make_snake_case(place_name)
    directory_name = f"{HEAT_OUTPUTS_PATH}{snake_case_place_name}/"

    if data_exists(GOOGLE_CLOUD_BUCKET, directory_name):
        print(f"Predictions data already exists for {place_name}. Skipping prediction.")
        return

    print("Processing data to classify...")
    bbox = get_area_of_interest(place_name)
    image_to_classify = process_data_to_classify(bbox)
    classified_image = classify_image(
        image_to_classify, HEAT_INPUT_PROPERTIES, HEAT_MODEL_ASSET_ID
    )

    predicted_image_filename = f"predicted_heat_hazard_{snake_case_place_name}"
    task = export_predictions(
        classified_image, predicted_image_filename, bucket, directory_name, HEAT_SCALE
    )

    monitor_tasks([task], 600)
