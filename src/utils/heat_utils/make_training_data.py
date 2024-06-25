import ee
import geemap
from data_utils.monitor_tasks import monitor_tasks
from data_utils.pygeoboundaries import get_adm_ee
from data_utils.export_and_monitor import start_export_task
from data_utils.scaling_factors import apply_scale_factors
from data_utils.cloud_mask import cloud_mask
from data_utils.export_ndvi import export_ndvi_min_max
from data_utils.download_ndvi import download_ndvi_data_for_year
from data_utils.process_annual_data import process_year
from data_utils.process_data_to_classify import process_data_to_classify
from google.cloud import storage
from datetime import datetime
import csv
from io import StringIO
from collections import Counter

import pretty_errors

# check if data for a country exists; if it does, skip with message; if not, proceed

# get images for a country

# write to GCS

# finish


def process_heat_data(place_name):

    cloud_project = "hotspotstoplight"
    ee.Initialize(project=cloud_project)

    current_year = datetime.now().year

    # Define the range for the previous 5 full calendar years
    years = range(current_year - 6, current_year - 1)

    scale = 90

    snake_case_place_name = place_name.replace(" ", "_").lower()

    aoi = get_adm_ee(territories=place_name, adm="ADM0")
    bbox = aoi.geometry().bounds()

    bucket_name = f"hotspotstoplight_heatmapping"
    directory_name = f"data/{snake_case_place_name}/inputs/"

    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    file_prefix = "ndvi_min_max"

    gcs_bucket = bucket_name

    def process_for_year(year, cloud_project, bucket_name, snake_case_place_name):

        ndvi_min, ndvi_max = download_ndvi_data_for_year(
            year, cloud_project, bucket_name, snake_case_place_name
        )
        image_collection = process_year(year, bbox, ndvi_min, ndvi_max)

        return image_collection

    for year in years:
        export_ndvi_min_max(year, bbox, scale, gcs_bucket, snake_case_place_name)

    image_list = []

    for year in years:
        image = process_for_year(
            year, cloud_project, bucket_name, snake_case_place_name
        )
        image_list.append(image)

    image_collections = ee.ImageCollection.fromImages(image_list)
