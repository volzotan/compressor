# aligner
INPUT_DIR_ALIGNER       = "/Users/volzotan/GIT/compressor/images_jpegs"
TRANSLATION_DATA        = "translation_data.json"
OUTPUT_DIR_ALIGNER      = "aligned"

REFERENCE_IMAGE         = "/Users/volzotan/Desktop/export_tiff/DSC03660.tif"

RESET_MATRIX_EVERY_LOOP = False
DOWNSIZE                = True
SKIP_TRANSLATION        = 10

JSON_SAVE_INTERVAL      = 10

# stacker
NAMING_PREFIX           = ""
INPUT_DIR_STACKER       = "~/Desktop/peakingtest"
OUTPUT_DIR_STACKER      = "/Users/volzotan/GIT/compressor/output"
FIXED_OUTPUT_NAME       = "frontlaan_peaking_value0.4.jpg"
EXTENSION               = ".jpg"
PICKLE_NAME             = "stack.pickle"

ALIGN                   = False

DISPLAY_CURVE           = False
APPLY_CURVE             = False

APPLY_PEAKING           = True
PEAKING_THRESHOLD       = 250
PEAKING_MUL_FACTOR      = 0.4

WRITE_METADATA          = True            
SORT_IMAGES             = False

SAVE_INTERVAL           = 5
PICKLE_INTERVAL         = -1

# debug

DISPLAY_PEAKING         = False

# meta
DIRS_TO_EXPAND          = [ "INPUT_DIR_ALIGNER", 
                            "OUTPUT_DIR_ALIGNER", 
                            "REFERENCE_IMAGE",
                            "TRANSLATION_DATA",
                            "INPUT_DIR_STACKER", 
                            "OUTPUT_DIR_STACKER",
                            ]

# DIRS_ABORT_IF_MISSING   = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]
DIRS_ABORT_IF_MISSING   = ["INPUT_DIR_STACKER"]
DIRS_TO_CREATE          = ["OUTPUT_DIR_ALIGNER", "OUTPUT_DIR_STACKER"]