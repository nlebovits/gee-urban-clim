LANDCOVER_SCALE = 10

HEAT_SCALE = 90
HEAT_INPUT_PROPERTIES = ["longitude", "latitude", "landcover", "elevation"]
HEAT_OUTPUTS_PATH = "heat/outputs/"
HEAT_INPUTS_PATH = "heat/inputs/"

FLOOD_SCALE = 30
FLOOD_INPUT_PROPERTIES = [
    "elevation",
    "landcover",
    "slope",
    "flow_direction",
    "stream_distance",
    "flow_accumulation",
    "spi",
    "sti",
    "cti",
    "tpi",
    "tri",
    "pcurv",
    "tcurv",
    "aspect",
]
FLOOD_OUTPUTS_PATH = "flood/ouputs/"
FLOOD_INPUTS_PATH = "flood/inputs/"
