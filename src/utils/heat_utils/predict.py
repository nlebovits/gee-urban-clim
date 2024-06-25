import ee


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