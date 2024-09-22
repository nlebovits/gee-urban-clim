import json
import os
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO

import ee
import numpy as np
import pandas as pd
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
    classify_image,
    data_exists,
    export_predictions,
    initialize_storage_client,
    monitor_tasks,
)

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")

ee.Initialize(project=GOOGLE_CLOUD_PROJECT)

samples_per_flood_class = 50000


def filter_data_from_gcs(country_name):
    """
    Pulls data from an Excel file in a Google Cloud Storage bucket,
    filters it based on a specified country name (case-insensitive) and years >= 2016,
    and returns the filtered data.

    Parameters:
    - country_name: The country name to filter the data by

    Returns:
    - A list of tuples with the start and end dates for the filtered rows
    """

    client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = client.bucket(GOOGLE_CLOUD_BUCKET)
    blob = bucket.blob(EMDAT_DATA_PATH)

    content = blob.download_as_bytes()

    excel_data = pd.read_excel(BytesIO(content), engine="openpyxl")

    filtered_data = excel_data[
        excel_data["Country"].str.lower() == country_name.lower()
    ].copy()  # Ensure this is a copy to avoid SettingWithCopyWarning

    for date_type in ["Start", "End"]:
        year_col = f"{date_type} Year"
        month_col = f"{date_type} Month"
        day_col = f"{date_type} Day"
        date_col = f"{date_type.lower()}_date"

        combined_dates = pd.to_datetime(
            {
                "year": filtered_data[year_col],
                "month": filtered_data[month_col],
                "day": filtered_data[day_col],
            },
            errors="coerce",
        )

        invalid_rows = filtered_data[combined_dates.isna()]
        if not invalid_rows.empty:
            print(f"Invalid {date_type.lower()} dates detected:")
            print(invalid_rows[[year_col, month_col, day_col]])

        filtered_data.loc[:, date_col] = combined_dates  # Use .loc to avoid warnings

    valid_data = filtered_data.dropna(subset=["start_date", "end_date"])

    valid_data = valid_data[valid_data["Start Year"] >= 2016]

    date_pairs = [
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        for start_date, end_date in zip(
            valid_data["start_date"], valid_data["end_date"]
        )
    ]

    return date_pairs


def make_training_data(bbox, start_date, end_date):
    before_start = (start_date - timedelta(days=10)).strftime("%Y-%m-%d")
    before_end = start_date.strftime("%Y-%m-%d")

    after_start = end_date.strftime("%Y-%m-%d")
    after_end = (end_date + timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"Generating training data for {start_date} to {end_date}...")

    dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(bbox)
    slope = ee.Terrain.slope(dem)
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    flow_direction = ee.Image("WWF/HydroSHEDS/03DIR").clip(bbox)

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

    # SET SAR PARAMETERS (can be left default)

    # Polarization (choose either "VH" or "VV")
    polarization = "VH"  # or "VV"

    # Difference threshold to be applied on the difference image (after flood - before flood)
    # It has been chosen by trial and error. Adjust as needed.
    difference_threshold = 1.25

    # Relative orbit (optional, if you know the relative orbit for your study area)
    # relative_orbit = 79

    # Load and filter Sentinel-1 GRD data by predefined parameters
    # Define helper function to check collection size
    def check_collection_size(pass_dir):
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", polarization)
            )
            .filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filterBounds(bbox)
            .select(polarization)
        )

        pre_collection_size = (
            collection.filterDate(before_start, before_end).size().getInfo()
        )
        post_collection_size = (
            collection.filterDate(after_start, after_end).size().getInfo()
        )

        return pre_collection_size > 0 and post_collection_size > 0, collection

    # Check both descending and ascending pass directions
    descending_available, descending_collection = check_collection_size("DESCENDING")
    ascending_available, ascending_collection = check_collection_size("ASCENDING")

    # Determine which collection to use
    if descending_available:
        collection = descending_collection
        print("Using DESCENDING pass direction.")
    elif ascending_available:
        collection = ascending_collection
        print("Using ASCENDING pass direction.")
    else:
        print("No pre-event or post-event imagery available.")
        return None

    # Select images by predefined dates
    before_collection = collection.filterDate(before_start, before_end)
    after_collection = collection.filterDate(after_start, after_end)

    # Check for imagery availability
    if before_collection.size().getInfo() == 0:
        print(
            f"No pre-event imagery available for the selected region and date range: {before_start} to {before_end}"
        )
        return None  # Exit the function early

    if after_collection.size().getInfo() == 0:
        print(
            f"No post-event imagery available for the selected region and date range: {after_start} to {after_end}"
        )
        return None  # Exit the function early

    # Create a mosaic of selected tiles and clip to the study area
    before = before_collection.mosaic().clip(bbox)
    after = after_collection.mosaic().clip(bbox)

    # Apply radar speckle reduction by smoothing
    smoothing_radius = 50
    before_filtered = before.focal_mean(smoothing_radius, "circle", "meters")
    after_filtered = after.focal_mean(smoothing_radius, "circle", "meters")

    # Calculate the difference between the before and after images
    difference = after_filtered.divide(before_filtered)

    # Apply the predefined difference-threshold and create the flood extent mask
    threshold = difference_threshold
    difference_binary = difference.gt(threshold)

    # Refine the flood result using additional datasets
    swater = ee.Image("JRC/GSW1_0/GlobalSurfaceWater").select("seasonality")
    swater_mask = swater.gte(10).updateMask(swater.gte(10))
    flooded_mask = difference_binary.where(swater_mask, 0)
    flooded = flooded_mask.updateMask(flooded_mask)
    connections = flooded.connectedPixelCount()
    flooded = flooded.updateMask(connections.gte(8))

    # Mask out areas with more than 5 percent slope using a Digital Elevation Model
    flooded = flooded.updateMask(slope.lt(5))

    hydro_proj = stream_dist_proximity.projection()

    # Set the default projection from the hydrography dataset
    flooded = flooded.setDefaultProjection(hydro_proj)

    # Create a full-area mask, initially marking everything as non-flooded (value 0)
    full_area_mask = ee.Image.constant(0).clip(bbox)

    # Update the mask to mark flooded areas (value 1)
    # Assuming flooded_mode is a binary image with 1 for flooded areas and 0 elsewhere
    flood_labeled_image = full_area_mask.where(flooded, 1)

    # Now flood_labeled_image contains 1 for flooded areas and 0 for non-flooded areas

    combined = (
        dem.toFloat()
        .rename("elevation")
        .addBands(landcover.select("Map").toFloat().rename("landcover"))
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
        .addBands(flood_labeled_image.toFloat().rename("flooded_mask"))
    )

    return combined


def generate_and_export_training_data():
    for country in TRAINING_DATA_COUNTRIES:
        print(f"Generating flood training data for {country}...")

        snake_case_place_name = country.replace(" ", "_").lower()

        # Define the base directory name
        base_directory = f"{FLOOD_INPUTS_PATH}{snake_case_place_name}/"

        # Check if flood training data already exists
        if data_exists(GOOGLE_CLOUD_BUCKET, f"{base_directory}flood_training_data_"):
            print(f"Flood training data already exists for {country}. Skipping...")
            continue

        # Get bounding box and filter data
        bbox = get_area_of_interest(country)
        date_pairs = filter_data_from_gcs(country)

        # Prepare date pairs for processing
        flood_dates = [
            (
                datetime.strptime(start, "%Y-%m-%d").date(),
                datetime.strptime(end, "%Y-%m-%d").date(),
            )
            for start, end in date_pairs
        ]

        # Initialize Google Cloud Storage client and create the new directory
        storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
        bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)
        blob = bucket.blob(base_directory)
        blob.upload_from_string(
            "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
        )

        tasks = []

        # Generate training data for each flood date range
        for start_date, end_date in flood_dates:
            combined = make_training_data(bbox, start_date, end_date)
            if combined:
                file_name = (
                    f"{base_directory}flood_training_data_{start_date}_{end_date}.tif"
                )
                task = ee.batch.Export.image.toCloudStorage(
                    image=combined,
                    description=f"{snake_case_place_name}_flood_training_data_{start_date}_{end_date}",
                    bucket=GOOGLE_CLOUD_BUCKET,
                    fileNamePrefix=file_name,
                    scale=FLOOD_SCALE,
                    region=bbox,
                    maxPixels=1e12,
                )
                task.start()
                tasks.append(task)

        monitor_tasks(tasks, 600)


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
    """Aggregate samples based on flooded status and export as a GeoJSON."""

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


def export_samples_to_gcs(samples, GOOGLE_CLOUD_BUCKET, filename):
    """Export samples to Google Cloud Storage as GeoJSON."""
    try:
        task = ee.batch.Export.table.toCloudStorage(
            collection=samples,
            description="ExportToGCS",
            bucket=GOOGLE_CLOUD_BUCKET,
            fileNamePrefix=filename,
            fileFormat="GeoJSON",
        )
        task.start()
        print(
            f"Export task {task.id} started, exporting samples to gs://{GOOGLE_CLOUD_BUCKET}/{filename}"
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


def read_geojson_from_gcs(GOOGLE_CLOUD_BUCKET, filename):
    """Read GeoJSON file from GCS, parse, clean, and convert into ee.FeatureCollection."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)
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
    accuracyMatrix, description, GOOGLE_CLOUD_BUCKET, filePrefix
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
        bucket=GOOGLE_CLOUD_BUCKET,
        fileNamePrefix=f"{filePrefix}",
        fileFormat="CSV",
    )
    task.start()
    print(
        f"Export task {task.id} started, exporting results to gs://{GOOGLE_CLOUD_BUCKET}/{filePrefix}/{description}.csv"
    )


def train_and_evaluate_classifier(
    image_collection, bbox, GOOGLE_CLOUD_BUCKET, snake_case_place_name
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

        flooded_data = flooded_data.randomColumn()
        unflooded_data = unflooded_data.randomColumn()

        train_split = 0.6
        test_split = 0.2

        flooded_training = flooded_data.filter(ee.Filter.lt("random", train_split))
        flooded_remaining = flooded_data.filter(ee.Filter.gte("random", train_split))

        flooded_testing = flooded_remaining.filter(
            ee.Filter.lt("random", train_split + test_split)
        )
        flooded_validation = flooded_remaining.filter(
            ee.Filter.gte("random", train_split + test_split)
        )

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

        training_samples = flooded_training.merge(unflooded_training)
        testing_samples = flooded_testing.merge(unflooded_testing)
        validation_samples = flooded_validation.merge(unflooded_validation)

        if not training_samples or not testing_samples or not validation_samples:
            print("Error: Failed to sample datasets.")
            return None, None, None

        print("Training the classifier...")
        classifier = ee.Classifier.smileRandomForest(10).train(
            features=training_samples,
            classProperty="flooded_mask",
            inputProperties=FLOOD_INPUT_PROPERTIES,
        )

        print("Evaluating the classifier...")
        test_accuracy = testing_samples.classify(classifier).errorMatrix(
            "flooded_mask", "classification"
        )
        validation_accuracy = validation_samples.classify(classifier).errorMatrix(
            "flooded_mask", "classification"
        )

        base_directory = (
            f"{FLOOD_OUTPUTS_PATH}{FLOOD_MODEL_ASSET_ID}/{snake_case_place_name}/"
        )

        print("Exporting results...")
        export_results_to_cloud_storage(
            test_accuracy,
            "Testing",
            GOOGLE_CLOUD_BUCKET,
            f"{base_directory}testing_results",
        )
        export_results_to_cloud_storage(
            validation_accuracy,
            "Validation",
            GOOGLE_CLOUD_BUCKET,
            f"{base_directory}validation_results",
        )

        print("Training probability predictor...")
        prob_classifier = (
            ee.Classifier.smileRandomForest(10)
            .setOutputMode("raw")
            .train(
                features=training_samples,
                classProperty="flooded_mask",
                inputProperties=FLOOD_INPUT_PROPERTIES,
            )
        )

        print("Training and evaluation process completed.")
        return prob_classifier, test_accuracy, validation_accuracy
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


def list_gcs_files(GOOGLE_CLOUD_BUCKET, prefix):
    """List all files in a GCS bucket folder."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(GOOGLE_CLOUD_BUCKET, prefix=prefix)
    return [
        f"gs://{GOOGLE_CLOUD_BUCKET}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]


def read_images_into_collection(GOOGLE_CLOUD_BUCKET, prefix):
    """Read images from cloud bucket into an Earth Engine image collection."""
    tif_list = list_gcs_files(GOOGLE_CLOUD_BUCKET, prefix)
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

    for country in TRAINING_DATA_COUNTRIES:
        print(f"Processing flood data for {country}...")

        snake_case_place_name = country.replace(" ", "_").lower()
        directory_name = f"{FLOOD_INPUTS_PATH}{snake_case_place_name}/"

        if not data_exists(GOOGLE_CLOUD_BUCKET, directory_name):
            print(f"No training data found for {country}. Skipping...")
            continue

        print(f"Reading images for {country} into collection...")
        country_image_collection = read_images_into_collection(
            GOOGLE_CLOUD_BUCKET, directory_name
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

    bbox = get_area_of_interest(TRAINING_DATA_COUNTRIES)

    prob_classifier, test_accuracy, validation_accuracy = train_and_evaluate_classifier(
        combined_image_collection, bbox, GOOGLE_CLOUD_BUCKET, "combined_model"
    )
    if prob_classifier is None:
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

    task = export_model_as_ee_asset(prob_classifier, FLOOD_MODEL_ASSET_ID)
    monitor_tasks([task], 60)

    print("Process completed successfully.")


def process_data_to_classify(bbox):
    dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(bbox)
    slope = ee.Terrain.slope(dem)
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    flow_direction = ee.Image("WWF/HydroSHEDS/03DIR").clip(bbox)

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


def predict(place_name: str) -> None:
    """
    Main function to predict flood risk for a given place and export the result.

    Args:
        place_name (str): The name of the place for which flood risk is predicted.
    """
    snake_case_place_name: str = place_name.replace(" ", "_").lower()
    base_directory: str = f"{FLOOD_OUTPUTS_PATH}{snake_case_place_name}/"
    predicted_image_filename: str = f"predicted_flood_risk_{snake_case_place_name}"

    if data_exists(GOOGLE_CLOUD_BUCKET, f"{base_directory}{predicted_image_filename}"):
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

    bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)
    task = export_predictions(
        classified_image, place_name, bucket, base_directory, FLOOD_SCALE
    )

    monitor_tasks([task], 600)
