import stacker

# Image file extension, either .jpg or .tif
# None: Autodetection
EXTENSION                       = None

# Try to sort image files by filename
SORT_IMAGES                     = True

# ----------------------- aligner -----------------------

# INPUT_DIR_ALIGNER               = "/Users/volzotan/Documents/DESPATDATASETS/20-01-22_marktplatz/frames"
INPUT_DIR_ALIGNER               = "/Users/volzotan/Documents/trashcam_library/2020-04-05_marktplatz/captures_1"
OUTPUT_DIR_ALIGNER              = INPUT_DIR_ALIGNER + "_aligned"
# OUTPUT_DIR_ALIGNER              = "/media/internal/DESPATDATASETS/20-01-26_theaterplatz" + "_aligned"
TRANSLATION_DATA                = OUTPUT_DIR_ALIGNER + "/" + "translation_data.json"

REFERENCE_IMAGE                 = INPUT_DIR_ALIGNER + "/cap_000385.jpg" #None 

RESET_MATRIX_EVERY_LOOP         = False
DOWNSIZE                        = True
DOWNSIZE_FACTOR                 = 2.0
SKIP_TRANSLATION                = 8 # do calculate translation data only from every n-th image

TRANSFER_METADATA               = False # copy metadata to transformed image file

JSON_SAVE_INTERVAL              = 200

# ----------------------- stitcher ----------------------

INPUT_DIR_STITCHER              = INPUT_DIR_ALIGNER
OUTPUT_DIR_STITCHER             = INPUT_DIR_STITCHER + "_stitch"

# ----------------------- stacker -----------------------


NAMING_PREFIX                   = ""
# INPUT_DIR_STACKER               = "/media/internal/DESPATDATASETS/20-01-17_goetheplatz_aligned"
# INPUT_DIR_STACKER               = "/Users/volzotan/Documents/DESPATDATASETS/19-12-21_augustbaudertplatz_aligned"
# INPUT_DIR_STACKER               = "/Volumes/ctdrive/export_theaterplatz5_tiff"
# INPUT_DIR_STACKER               = "/media/internal/DESPATDATASETS/20-01-26_theaterplatz/frames"
INPUT_DIR_STACKER               = "/Users/volzotan/Documents/trashcam_library/2020-07-06_pragerstrasse1/captures_1"
# INPUT_DIR_STACKER               = "/Users/volzotan/Downloads/hqtest/captures_1"
# OUTPUT_DIR_STACKER              = "/media/internal/DESPATDATASETS/20-01-26_theaterplatz_aligned_stacked"
OUTPUT_DIR_STACKER              = INPUT_DIR_STACKER + "_stacked" 
# OUTPUT_DIR_STACKER              = "/Users/volzotan/Downloads/stackeroutput/output9_stacked" 
FIXED_OUTPUT_NAME               = "output"
PICKLE_NAME                     = "stack.pickle"

ALIGN                           = False

# minimum value for average image brightness for input data filtering
MIN_BRIGHTNESS_THRESHOLD        = 10

APPLY_CURVE                     = True

APPLY_PEAKING                   = False
PEAKING_STRATEGY                = stacker.PEAKING_MODE_LIGHTEN
PEAKING_FROM_2ND_IMAGE          = True 
PEAKING_IMAGE_THRESHOLD         = None #11
PEAKING_BLEND                   = False
PEAKING_PIXEL_THRESHOLD         = None  # is ignored right now
PEAKING_MUL_FACTOR              = 0.3   # is ignored right now

WRITE_METADATA                  = True

SAVE_INTERVAL                   = 100
INTERMEDIATE_SAVE_FORCE_JPEG    = False # still buggy (values to high for jpeg saving)
PICKLE_INTERVAL                 = -1

# debug

DISPLAY_CURVE                   = False
DISPLAY_PEAKING                 = False


# ----------------------- meta --------------------------

DIRS_TO_EXPAND_ALIGNER          = [ "INPUT_DIR_ALIGNER", 
                                    "OUTPUT_DIR_ALIGNER", 
                                    "REFERENCE_IMAGE",
                                    "TRANSLATION_DATA"
                                    ]
DIRS_TO_EXPAND_STITCHER         = []
DIRS_TO_EXPAND_STACKER          = [ "TRANSLATION_DATA",
                                    "INPUT_DIR_STACKER", 
                                    "OUTPUT_DIR_STACKER",
                                    ]

DIRS_ABORT_IF_MISSING_ALIGNER   = ["INPUT_DIR_ALIGNER"]
DIRS_ABORT_IF_MISSING_STITCHER  = ["INPUT_DIR_STITCHER"]
DIRS_ABORT_IF_MISSING_STACKER   = ["INPUT_DIR_STACKER"]
DIRS_TO_CREATE_ALIGNER          = ["OUTPUT_DIR_ALIGNER"]
DIRS_TO_CREATE_STITCHER         = ["OUTPUT_DIR_STITCHER"]
DIRS_TO_CREATE_STACKER          = ["OUTPUT_DIR_STACKER"]