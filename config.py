# Image file extension, either .jpg or .tif
# None: Autodetection
EXTENSION                       = None #autodetection

# Try to sort image files by filename
SORT_IMAGES                     = True

# ----------------------- aligner -----------------------

INPUT_DIR_ALIGNER               = "/Users/volzotan/Downloads/export_bangkok120_jpeg_unaligned"
# INPUT_DIR_ALIGNER               = "/Users/volzotan/GIT/stacktest/Z_ZUGSPITZE/cropped"
TRANSLATION_DATA                = "translation_data.json"
OUTPUT_DIR_ALIGNER              = INPUT_DIR_ALIGNER + "_aligned"

REFERENCE_IMAGE                 = None #"~/Desktop/peakingtest/DSC04370.jpg"

RESET_MATRIX_EVERY_LOOP         = False
DOWNSIZE                        = False
DOWNSIZE_FACTOR                 = 2.0
SKIP_TRANSLATION                = -1

JSON_SAVE_INTERVAL              = 100

# ----------------------- stitcher ----------------------

INPUT_DIR_STITCHER              = INPUT_DIR_ALIGNER
OUTPUT_DIR_STITCHER             = INPUT_DIR_STITCHER + "_stitch"

# ----------------------- stacker -----------------------

NAMING_PREFIX                   = ""
INPUT_DIR_STACKER               = "/Users/volzotan/Downloads/export_bangkok120_jpeg_unaligned"
# INPUT_DIR_STACKER               = "/Volumes/ctdrive/export_hongkong4"
OUTPUT_DIR_STACKER              = INPUT_DIR_STACKER + "_stacked" 
FIXED_OUTPUT_NAME               = "output"
PICKLE_NAME                     = "stack.pickle"

ALIGN                           = True

APPLY_CURVE                     = True

APPLY_PEAKING                   = True
PEAKING_STRATEGY                = "lighten"
PEAKING_FROM_2ND_IMAGE          = True 
PEAKING_IMAGE_THRESHOLD         = 12
PEAKING_BLEND                   = False
PEAKING_PIXEL_THRESHOLD         = None  # is ignored right now
PEAKING_MUL_FACTOR              = 0.3   # is ignored right now

WRITE_METADATA                  = True

SAVE_INTERVAL                   = 5
INTERMEDIATE_SAVE_FORCE_JPEG    = False # still buggy (values to high for jpeg saving)
PICKLE_INTERVAL                 = -1

# debug

DISPLAY_CURVE                   = True
DISPLAY_PEAKING                 = False

# meta
DIRS_TO_EXPAND_ALIGNER          = [ "INPUT_DIR_ALIGNER", 
                                    "OUTPUT_DIR_ALIGNER", 
                                    "REFERENCE_IMAGE",
                                    "TRANSLATION_DATA"
                                    ]
DIRS_TO_EXPAND_STACKER          = [ "TRANSLATION_DATA",
                                    "INPUT_DIR_STACKER", 
                                    "OUTPUT_DIR_STACKER",
                                    ]

DIRS_ABORT_IF_MISSING_ALIGNER   = ["INPUT_DIR_ALIGNER"]
DIRS_ABORT_IF_MISSING_STACKER   = ["INPUT_DIR_STACKER"]
DIRS_TO_CREATE_ALIGNER          = ["OUTPUT_DIR_ALIGNER"]
DIRS_TO_CREATE_STACKER          = ["OUTPUT_DIR_STACKER"]