import ee
import csv
from google.cloud import storage
from io import StringIO

import ee
from data_utils.cloud_mask import cloud_mask
from data_utils.scaling_factors import apply_scale_factors



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