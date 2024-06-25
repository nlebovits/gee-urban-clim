import csv
from collections import Counter
from datetime import datetime
from io import StringIO

import ee
import geemap
import pretty_errors
from data_utils.cloud_mask import cloud_mask
from data_utils.download_ndvi import download_ndvi_data_for_year
from data_utils.export_and_monitor import start_export_task
from data_utils.export_ndvi import export_ndvi_min_max
from data_utils.monitor_tasks import monitor_tasks
from data_utils.process_annual_data import process_year
from data_utils.process_data_to_classify import process_data_to_classify
from data_utils.pygeoboundaries import get_adm_ee
from data_utils.scaling_factors import apply_scale_factors
from google.cloud import storage

# check if data for a country exists; if it does, skip with message; if not, proceed

# get images for a country

# write to GCS

# finish


## needed inputs:

    # list of place names
    # cloud project as GOOGLE_CLOUD_PROJECT from .env file
    # bucket name as GOOGLE_CLOUD_BUCKET from .env file
    # HEAT_SCALE from config.py (set as 90m)

## needed steps:

    # print list of countries for which it's generating data
    # one by one:
        # check if data exists for a given country
        # if it does, skip the country w message
        # if it doesn't, print a message saying that it's starting to generate the data
        # generate data for a given country; save images to GCS bucket with appropriate file name
        # move on to next country
        # finish when list is complete

def process_heat_data(place_name):

    cloud_project = GOOGLE_CLOUD_PROJECT
    ee.Initialize(project=cloud_project)

    current_year = datetime.now().year

    # Define the range for the previous 5 full calendar years
    years = range(current_year - 6, current_year - 1)

    scale = 90

    snake_case_place_name = place_name.replace(" ", "_").lower()

    aoi = get_adm_ee(territories=place_name, adm="ADM0")
    bbox = aoi.geometry().bounds()

    bucket_name = GOOGLE_CLOUD_BUCKET
    directory_name = f"heat_data/{snake_case_place_name}/inputs/"

    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    def process_for_year(year, cloud_project, bucket_name, snake_case_place_name):

        ndvi_min, ndvi_max = download_ndvi_data_for_year(
            year, cloud_project, bucket_name, snake_case_place_name
        )
        image_collection = process_year(year, bbox, ndvi_min, ndvi_max)

        return image_collection

    for year in years:
        export_ndvi_min_max(year, bbox, scale, bucket_name snake_case_place_name)

    image_list = []

    for year in years:
        image = process_for_year(
            year, cloud_project, bucket_name, snake_case_place_name
        )

        # write the image to GCS with the directory name plus the year in the path (print the path while writing)



def process_year(year, bbox, ndvi_min, ndvi_max):
    # Define the start and end dates for the year
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)

    # Import and preprocess Landsat 8 imagery for the year
    imageCollection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(bbox)
        .filterDate(startDate, endDate)
        .map(apply_scale_factors)
        .map(cloud_mask)
    )

    # Function to calculate LST for each image in the collection
    def calculate_lst(image):
        # Calculate Normalized Difference Vegetation Index (NDVI)
        ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

        # Use the passed ndvi_min and ndvi_max directly instead of calculating them
        # Convert them to ee.Number since they are likely passed as Python primitives
        ndvi_min_ee = ee.Number(ndvi_min)
        ndvi_max_ee = ee.Number(ndvi_max)

        # Fraction of Vegetation (FV) Calculation
        fv = (
            ee.Image()
            .expression(
                "(ndvi - ndvi_min) / (ndvi_max - ndvi_min)",
                {"ndvi": ndvi, "ndvi_max": ndvi_max_ee, "ndvi_min": ndvi_min_ee},
            )
            .pow(2)
            .rename("FV")
        )

        # Emissivity Calculation
        em = fv.multiply(ee.Number(0.004)).add(ee.Number(0.986)).rename("EM")

        # Select Thermal Band (Band 10) and Rename It
        thermal = image.select("ST_B10").rename("thermal")

        # Land Surface Temperature (LST) Calculation
        lst = thermal.expression(
            "(TB / (1 + (0.00115 * (TB / 1.438)) * log(em))) - 273.15",
            {
                "TB": thermal.select("thermal"),  # Select the thermal band
                "em": em,  # Assign emissivity
            },
        ).rename("LST")

        return lst

    # Apply the calculate_lst function to each image in the collection
    lstCollection = imageCollection.map(calculate_lst)

    # Create a binary image for each image in the collection where 1 indicates LST >= 33 and 0 otherwise
    hotDaysCollection = lstCollection.map(lambda image: image.gte(33))

    # Sum all the binary images in the collection to get the total number of hot days in the year
    hotDaysYear = hotDaysCollection.sum()
    
    # return a band that represents the median value of the top 5 hottest days
    hotDaysYear = hotDaysYear.rename("hot_days")
    
    # Convert the collection of LST images into an array.
    array = lstCollection.toArray()

    # Sort the array in descending order (highest LST values first).
    sortedArray = array.arraySort().arraySlice(0, -5)  # Slice to get the last 5 elements, which are the top 5 when sorted in descending order.

    # Calculate the median of the top 5 values for each pixel.
    medianOfTop5 = sortedArray.arrayReduce(ee.Reducer.median(), [0]).arrayProject([1]).arrayFlatten([['median']])

    
    # Get the landcover, elevation, a

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

import csv
from io import StringIO

import ee
from data_utils.cloud_mask import cloud_mask
from data_utils.scaling_factors import apply_scale_factors
from google.cloud import storage


def download_ndvi_data_for_year(
    year, cloud_project, bucket_name, snake_case_place_name
):
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client(project=cloud_project)
    bucket = storage_client.bucket(bucket_name)

    # Define the blob's name to include the full path
    blob_name = f"data/{snake_case_place_name}/inputs/ndvi_min_max_{year}.csv"
    blob = bucket.blob(blob_name)

    # Download the data as a string
    ndvi_data_csv = blob.download_as_string()

    # Parse the CSV data
    ndvi_data = csv.reader(StringIO(ndvi_data_csv.decode("utf-8")))
    rows = list(ndvi_data)

    # Extract NDVI min and max values
    # Assuming the first row after the header contains NDVI min and the second row contains NDVI max
    # Note: This assumes row 1 is headers
    ndvi_min = float(rows[1][1])
    ndvi_max = float(rows[1][2])

    return ndvi_min, ndvi_max


def export_ndvi_min_max(
    year, bbox, scale, gcs_bucket, snake_case_place_name, file_prefix="ndvi_min_max"
):
    try:
        startDate = ee.Date.fromYMD(year, 1, 1)
        endDate = ee.Date.fromYMD(year, 12, 31)

        # Filter the collection for the given year and bounds
        imageCollection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(bbox)
            .filterDate(startDate, endDate)
            .map(apply_scale_factors)
            .map(cloud_mask)
        )

        # Calculate NDVI for the entire collection
        ndviCollection = imageCollection.map(
            lambda image: image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        )

        # Reduce the collection to get min and max NDVI values
        ndvi_min = ndviCollection.reduce(ee.Reducer.min()).reduceRegion(
            reducer=ee.Reducer.min(), geometry=bbox, scale=scale, maxPixels=1e9
        )
        ndvi_max = ndviCollection.reduce(ee.Reducer.max()).reduceRegion(
            reducer=ee.Reducer.max(), geometry=bbox, scale=scale, maxPixels=1e9
        )

        # Create a feature to export
        feature = ee.Feature(
            None,
            {
                "ndvi_min": ndvi_min.get("NDVI_min"),
                "ndvi_max": ndvi_max.get("NDVI_max"),
            },
        )

        # Create and start the export task with the specified fileNamePrefix
        task = ee.batch.Export.table.toCloudStorage(
            collection=ee.FeatureCollection([feature]),
            description=f"{file_prefix}_{year}",
            bucket=gcs_bucket,
            fileNamePrefix=f"data/{snake_case_place_name}/inputs/{file_prefix}_{year}",
            fileFormat="CSV",
        )
        task.start()

        # Print statements confirming the task has started
        print(f"Starting export task for NDVI min/max values of year {year}.")

        # Return the task object
        return task

    except Exception as e:
        print(f"An error occurred while starting the export task for year {year}: {e}")
        return None

# Applies scaling factors.
def apply_scale_factors(image):
    # Scale and offset values for optical bands
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)

    # Scale and offset values for thermal bands
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)

    # Add scaled bands to the original image
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


# Function to Mask Clouds and Cloud Shadows in Landsat 8 Imagery
def cloud_mask(image):
    # Define cloud shadow and cloud bitmasks (Bits 3 and 5)
    cloud_shadow_bitmask = 1 << 3
    cloud_bitmask = 1 << 5

    # Select the Quality Assessment (QA) band for pixel quality information
    qa = image.select("QA_PIXEL")

    # Create a binary mask to identify clear conditions (both cloud and cloud shadow bits set to 0)
    mask = (
        qa.bitwiseAnd(cloud_shadow_bitmask)
        .eq(0)
        .And(qa.bitwiseAnd(cloud_bitmask).eq(0))
    )

    # Update the original image, masking out cloud and cloud shadow-affected pixels
    return image.updateMask(mask)
