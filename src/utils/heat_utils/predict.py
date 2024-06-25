import ee

# Process data and classify the image
image_to_classify = process_data_to_classify(bbox)
classified_image = image_to_classify.select(inputProperties).classify(regressor)

# Initialize the storage client
storage_client = storage.Client(project=cloud_project)
bucket = storage_client.bucket(bucket_name)

# Create and upload an empty file to initialize the directory (if needed)
blob = bucket.blob(directory_name)
blob.upload_from_string(
    "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
)

# Export the predicted image
# Ensure the filename or path for the predicted image is unique to avoid overwriting
predicted_image_filename = f"predicted_median_top5_{snake_case_place_name}"  # Example filename, ensure it's unique
# The function `start_export_task` should handle the export logic, including setting the correct filename/path
task = start_export_task(
    classified_image,
    f"{place_name} Median temp of top 5 hottest observations",
    bucket_name,
    directory_name + predicted_image_filename,
    scale,
)
tasks = [task]
monitor_tasks(tasks, 600)


def process_data_to_classify(bbox):
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().clip(bbox)

    image_to_classify = (
        landcover.rename("landcover")
        .addBands(dem.rename("elevation"))
        .addBands(ee.Image.pixelLonLat())
    )

    print("Sampling image band names", image_to_classify.bandNames().getInfo())

    return image_to_classify
