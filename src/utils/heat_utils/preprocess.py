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