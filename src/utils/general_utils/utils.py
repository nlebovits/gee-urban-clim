# function to initialize google cloud storage connection-------------------------------------------------------
def initialize_storage_client(project, GOOGLE_CLOUD_BUCKET):
    """Initialize the Google Cloud Storage client."""
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)
    return bucket


def initialize_storage_client(project, bucket_name):
    """Initialize the Google Cloud Storage client."""
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    return bucket


# functions to start and monitor ee export tasks-------------------------------------------------------
import time

import ee


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


# function to get adm ee-------------------------------------------------------
def get_area_of_interest(place_name):
    """Retrieve the area of interest based on the place name."""
    return get_adm_ee(territories=place_name, adm="ADM0").geometry().bounds()


def get_area_of_interest(place_name):
    """Retrieve the area of interest based on the place name."""
    return get_adm_ee(territories=place_name, adm="ADM0").geometry().bounds()


# function to check if a file or files exist before proceeding-------------------------------------------------------
def list_gcs_files(GOOGLE_CLOUD_BUCKET, prefix):
    """List all files in a GCS bucket folder."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(GOOGLE_CLOUD_BUCKET, prefix=prefix)
    return [
        f"gs://{GOOGLE_CLOUD_BUCKET}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]


def list_blobs_with_prefix(bucket_name, prefix):
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    return list(bucket.list_blobs(prefix=prefix))


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


def extract_date_from_filename(filename):
    # Use a regular expression to find dates in the format YYYY-MM-DD
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    else:
        return None


# function for stratified sampling based on land cover classes-------------------------------------------------------


# function to read images in a directory from GCS into an image collection-------------------------------------------------------
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


def read_data_to_image_collection(TRAINING_DATA_COUNTRIES):
    bucket_name = GOOGLE_CLOUD_BUCKET
    all_tif_list = []

    # Check for data existence and collect URIs
    for country in TRAINING_DATA_COUNTRIES:
        snake_case_place_name = country.replace(" ", "_").lower()
        directory_name = f"{HEAT_INPUTS_PATH}{snake_case_place_name}/"

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


# function to export a trained classifier-------------------------------------------------------
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


# function to import a trained classifier and classify an image-------------------------------------------------------
def classify_image(image_to_classify, FLOOD_INPUT_PROPERTIES, model_asset_id):
    """Classify the image using the pre-trained model."""
    regressor = ee.Classifier.load(model_asset_id)
    return image_to_classify.select(FLOOD_INPUT_PROPERTIES).classify(regressor)


def classify_image(image_to_classify, input_properties, model_asset_id):
    """Classify the image using the pre-trained model."""
    regressor = ee.Classifier.load(model_asset_id)
    return image_to_classify.select(input_properties).classify(regressor)


# function to make predcitions-------------------------------------------------------
def predict(place_name):
    """Main function to predict flood risk for a given place and export the result."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    base_directory = f"{FLOOD_OUTPUTS_PATH}{snake_case_place_name}/"
    predicted_image_filename = f"predicted_flood_risk_{snake_case_place_name}"

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

    def process_data_to_classify(bbox):
        """Prepare the data to be classified."""
        landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
        dem = (
            ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
            .mosaic()
            .clip(bbox)
        )

        image_to_classify = (
            landcover.rename("landcover")
            .addBands(dem.rename("elevation"))
            .addBands(ee.Image.pixelLonLat())
        )

        return image_to_classify


def predict(place_name):
    """Main function to predict heat for a given place and export the result."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    directory_name = f"{HEAT_OUTPUTS_PATH}{snake_case_place_name}/"

    # Check if predictions data already exists
    if data_exists(GOOGLE_CLOUD_BUCKET, directory_name):
        print(f"Predictions data already exists for {place_name}. Skipping prediction.")
        return

    print("Processing data to classify...")
    bbox = get_area_of_interest(place_name)
    image_to_classify = process_data_to_classify(bbox)
    classified_image = classify_image(
        image_to_classify, HEAT_INPUT_PROPERTIES, HEAT_MODEL_ASSET_ID
    )

    # Initialize the storage client and export predictions
    bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)
    task = export_predictions(
        classified_image, place_name, bucket, directory_name, HEAT_SCALE
    )

    # Monitor the export task
    monitor_tasks([task], 600)


# function to export predictions-------------------------------------------------------
def export_predictions(classified_image, place_name, bucket, directory_name, scale):
    """Export the predictions to Google Cloud Storage."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    predicted_image_filename = f"predicted_flood_risk_{snake_case_place_name}"

    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    task = start_export_task(
        classified_image,
        f"{place_name} predicted flood risk",
        bucket.name,
        directory_name + predicted_image_filename,
        scale,
    )
    return task


def export_predictions(classified_image, place_name, bucket, directory_name, scale):
    """Export the predictions to Google Cloud Storage."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    predicted_image_filename = f"predicted_heat_hazard_{snake_case_place_name}"

    # Ensure the directory exists by uploading an empty file
    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    # Start export task
    task = start_export_task(
        classified_image,
        f"{place_name} predicted median temp of top 5 hottest observations",
        bucket.name,
        directory_name + predicted_image_filename,
        scale,
    )
    return task
