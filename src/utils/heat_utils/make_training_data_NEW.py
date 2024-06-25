import ee
import geemap
from utils.general_utils.monitor_ee_tasks import monitor_tasks
from utils.general_utils.pygeoboundaries import get_adm_ee
from utils.general_utils.monitor_ee_tasks import start_export_task
from google.cloud import storage
from datetime import datetime
import csv
from io import StringIO
from collections import Counter
import pretty_errors
import os
from dotenv import load_dotenv

from config import HEAT_SCALE

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


training_data_countries = ["Costa Rica", "Netherlands"]


# Check if data for a country exists; if it does, skip with message; if not, proceed
def data_exists(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


def process_heat_data(place_name):
    cloud_project = GOOGLE_CLOUD_PROJECT
    ee.Initialize(project=cloud_project)
    current_year = datetime.now().year
    years = range(current_year - 6, current_year - 1)
    scale = HEAT_SCALE
    snake_case_place_name = place_name.replace(" ", "_").lower()
    aoi = get_adm_ee(territories=place_name, adm="ADM0")
    bbox = aoi.geometry().bounds()
    bucket_name = GOOGLE_CLOUD_BUCKET
    directory_name = f"heat_data/{snake_case_place_name}/inputs/"

    if data_exists(bucket_name, directory_name):
        print(f"Data for {place_name} already exists. Skipping...")
        return
    else:
        print(f"Starting to generate data for {place_name}...")

    def process_for_year(year, cloud_project, bucket_name, snake_case_place_name):
        ndvi_min, ndvi_max = download_ndvi_data_for_year(
            year, cloud_project, bucket_name, snake_case_place_name
        )
        image_collection = process_year(year, bbox, ndvi_min, ndvi_max)
        return image_collection

    for year in years:
        export_ndvi_min_max(year, bbox, scale, bucket_name, snake_case_place_name)

    image_list = []

    for year in years:
        image = process_for_year(
            year, cloud_project, bucket_name, snake_case_place_name
        )
        image_list.append(image)

    for i, image in enumerate(image_list):
        year = years[i]
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=f"{snake_case_place_name}_heat_{year}",
            bucket=bucket_name,
            fileNamePrefix=f"{directory_name}{year}/heat_{year}",
            scale=scale,
            region=bbox.getInfo()["coordinates"],
        )
        task.start()
        print(f"Exporting heat data for {place_name} for the year {year} to GCS...")

    monitor_tasks()


def main(training_data_countries):

    print("Generating data for the following countries:")
    print(", ".join(training_data_countries))

    for place_name in training_data_countries:
        process_heat_data(place_name)

    print("Data generation completed.")


if __name__ == "__main__":
    main()


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


def download_ndvi_data_for_year(
    year, cloud_project, bucket_name, snake_case_place_name
):
    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)
    blob_name = f"data/{snake_case_place_name}/inputs/ndvi_min_max_{year}.csv"
    blob = bucket.blob(blob_name)
    ndvi_data_csv = blob.download_as_string()
    ndvi_data = csv.reader(StringIO(ndvi_data_csv.decode("utf-8")))
    rows = list(ndvi_data)
    ndvi_min = float(rows[1][1])
    ndvi_max = float(rows[1][2])
    return ndvi_min, ndvi_max


def export_ndvi_min_max(
    year, bbox, scale, gcs_bucket, snake_case_place_name, file_prefix="ndvi_min_max"
):
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
            bucket=gcs_bucket,
            fileNamePrefix=f"data/{snake_case_place_name}/inputs/{file_prefix}_{year}",
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
