import io
from datetime import datetime, timedelta

import ee
import pandas as pd
from google.cloud import storage

training_countries = ["Costa Rica"]


def filter_data_from_gcs(country_name):
    """
    Pulls data from an Excel file in a Google Cloud Storage bucket,
    filters it based on a specified country name (case-insensitive),
    and returns the filtered data.

    Parameters:
    - country_name: The country name to filter the data by

    Returns:
    - A list of tuples with the start and end dates for the filtered rows
    """
    bucket_name = "hotspotstoplight_floodmapping"
    file_name = "data/emdat/public_emdat_custom_request_2024-02-10_39ba89ea-de1d-4020-9b8e-027db50a5ded.xlsx"

    # Initialize a client and get the bucket and blob
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the blob into an in-memory file
    content = blob.download_as_bytes()

    # Read the Excel file into a DataFrame
    excel_data = pd.read_excel(io.BytesIO(content), engine="openpyxl")

    # Filter the DataFrame based on the 'Country' column, case-insensitive
    filtered_data = excel_data[
        excel_data["Country"].str.lower() == country_name.lower()
    ].copy()  # Ensure this is a copy to avoid SettingWithCopyWarning

    # Process start and end dates
    for date_type in ["Start", "End"]:
        year_col = f"{date_type} Year"
        month_col = f"{date_type} Month"
        day_col = f"{date_type} Day"
        date_col = f"{date_type.lower()}_date"

        # Combine the date components into a single date column
        combined_dates = pd.to_datetime(
            {
                "year": filtered_data[year_col],
                "month": filtered_data[month_col],
                "day": filtered_data[day_col],
            },
            errors="coerce",
        )

        # Detect rows where dates could not be parsed and print them
        invalid_rows = filtered_data[combined_dates.isna()]
        if not invalid_rows.empty:
            print(f"Invalid {date_type.lower()} dates detected:")
            print(invalid_rows[[year_col, month_col, day_col]])

        # Assign parsed dates back to the main DataFrame
        filtered_data.loc[:, date_col] = combined_dates  # Use .loc to avoid warnings

    # Filter out rows where either start_date or end_date are NaT
    valid_data = filtered_data.dropna(subset=["start_date", "end_date"])

    # Create date pairs as a list of tuples
    date_pairs = [
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        for start_date, end_date in zip(
            valid_data["start_date"], valid_data["end_date"]
        )
    ]

    return date_pairs


def make_training_data(bbox, start_date, end_date):

    # Convert the dates to datetime objects
    start_date = start_date
    end_date = end_date

    # Calculate the new dates
    before_start = (start_date - timedelta(days=10)).strftime("%Y-%m-%d")
    before_end = start_date.strftime("%Y-%m-%d")

    after_start = end_date.strftime("%Y-%m-%d")
    after_end = (end_date + timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"Generating training data for {start_date} to {end_date}...")

    year_before_start = start_date - timedelta(days=365)
    start_of_year = datetime(year_before_start.year, 1, 1)
    end_of_year = datetime(year_before_start.year, 12, 31)

    # Load the datasets

    dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(bbox)
    slope = ee.Terrain.slope(dem)
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    flow_direction = ee.Image("WWF/HydroSHEDS/03DIR").clip(bbox)
    ghsl = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018").clip(bbox)

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

    # Pass direction (choose either "DESCENDING" or "ASCENDING")
    pass_direction = "DESCENDING"  # or "ASCENDING"

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
        dem.rename("elevation")
        .addBands(landcover.select("Map").rename("landcover"))
        .addBands(slope)
        .addBands(ghsl)
        .addBands(flow_direction.rename("flow_direction"))
        .addBands(stream_dist_proximity)
        .addBands(flood_labeled_image.rename("flooded_mask"))
        .addBands(flow_accumulation)
        .addBands(spi)
        .addBands(sti)
        .addBands(cti)
        .addBands(tpi)
        .addBands(tri)
        .addBands(pcurv)
        .addBands(tcurv)
        .addBands(aspect)
    )

    return combined
