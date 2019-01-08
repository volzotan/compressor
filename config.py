EXTENSION               = None #autodetection

# ----------------------- aligner -----------------------

INPUT_DIR_ALIGNER       = "/Users/volzotan/Downloads/export_bangkok120_jpeg_unaligned"
TRANSLATION_DATA        = "translation_data.json"
OUTPUT_DIR_ALIGNER      = "aligned"

REFERENCE_IMAGE         = None #"~/Desktop/peakingtest/DSC04370.jpg"

RESET_MATRIX_EVERY_LOOP = False
DOWNSIZE                = False
DOWNSIZE_FACTOR         = 4.0
SKIP_TRANSLATION        = -1

JSON_SAVE_INTERVAL      = 100

# ----------------------- stacker -----------------------

# NAMING_PREFIX           = ""
# INPUT_DIR_STACKER       = "/Users/volzotan/Downloads/export_bangkok120_jpeg"
# OUTPUT_DIR_STACKER      = INPUT_DIR_STACKER + "_stacked" # "/Users/volzotan/Documents/DESPATDATASETS/stack"
# FIXED_OUTPUT_NAME       = "output"
# PICKLE_NAME             = "stack.pickle"

NAMING_PREFIX           = ""
INPUT_DIR_STACKER       = "/Users/volzotan/Downloads/export_hongkong4"
OUTPUT_DIR_STACKER      = INPUT_DIR_STACKER + "_stacked" # "/Users/volzotan/Documents/DESPATDATASETS/stack"
FIXED_OUTPUT_NAME       = "output"
PICKLE_NAME             = "stack.pickle"

ALIGN                   = False

APPLY_CURVE             = True

APPLY_PEAKING           = False
PEAKING_THRESHOLD       = -1    # auto
PEAKING_MUL_FACTOR      = 0.4

WRITE_METADATA          = True            
SORT_IMAGES             = True

SAVE_INTERVAL           = 50
INTERMEDIATE_SAVE_FORCE_JPEG  = False # still buggy (values to high for jpeg saving)
PICKLE_INTERVAL         = -1

# debug

DISPLAY_CURVE           = True
DISPLAY_PEAKING         = False

# meta
DIRS_TO_EXPAND_ALIGNER  = [ "INPUT_DIR_ALIGNER", 
                            "OUTPUT_DIR_ALIGNER", 
                            "REFERENCE_IMAGE",
                            "TRANSLATION_DATA"
                            ]
DIRS_TO_EXPAND_STACKER  = [ "TRANSLATION_DATA",
                            "INPUT_DIR_STACKER", 
                            "OUTPUT_DIR_STACKER",
                            ]

DIRS_ABORT_IF_MISSING_ALIGNER   = ["INPUT_DIR_ALIGNER"]
DIRS_ABORT_IF_MISSING_STACKER   = ["INPUT_DIR_STACKER"]
DIRS_TO_CREATE_ALIGNER          = ["OUTPUT_DIR_ALIGNER"]
DIRS_TO_CREATE_STACKER          = ["OUTPUT_DIR_STACKER"]