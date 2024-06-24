import re

import ee
from data_utils.make_training_data import make_training_data
from utils.general_utils.monitor_ee_tasks import monitor_tasks
from google.cloud import storage


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
    # print(tif_list)

    print("Reading images from cloud bucket into image collection...")
    ee_image_list = [ee.Image.loadGeoTIFF(url) for url in tif_list]
    image_collection = ee.ImageCollection.fromImages(ee_image_list)

    info = image_collection.size().getInfo()
    print(f"Collection contains {info} images.")

    return image_collection


def extract_date_from_filename(filename):
    # Use a regular expression to find dates in the format YYYY-MM-DD
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    else:
        return None

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
