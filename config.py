# aligner
INPUT_DIR_ALIGNER       = "/Users/volzotan/Desktop/export_tiff"
TRANSLATION_DATA        = "translation_data.json"
OUTPUT_DIR_ALIGNER      = "aligned"

REFERENCE_IMAGE         = "/Users/volzotan/Desktop/export_tiff/DSC03660.tif"

RESET_MATRIX_EVERY_LOOP = False
DOWNSIZE                = True
SKIP_TRANSLATION        = 10

JSON_SAVE_INTERVAL      = 10

# stacker
NAMING_PREFIX           = ""
INPUT_DIR_STACKER       = INPUT_DIR_ALIGNER
OUTPUT_DIR_STACKER      = "stack_bauhaus"
FIXED_OUTPUT_NAME       = "bauhaus.tif"
EXTENSION               = ".tif"
PICKLE_NAME             = "stack.pickle"

ALIGN                   = True
DISPLAY_CURVE           = False
APPLY_CURVE             = False

DISPLAY_PEAKING         = False
APPLY_PEAKING           = False
PEAKING_THRESHOLD       = 3000
PEAKING_MUL_FACTOR      = 1.0

WRITE_METADATA          = True
SORT_IMAGES             = False

SAVE_INTERVAL           = 30
PICKLE_INTERVAL         = -1

# meta
DIRS_TO_EXPAND          = [ "INPUT_DIR_ALIGNER", 
                            "OUTPUT_DIR_ALIGNER", 
                            "TRANSLATION_DATA",
                            "INPUT_DIR_STACKER", 
                            "OUTPUT_DIR_STACKER",
                            ]
DIRS_ABORT_IF_MISSING   = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]
DIRS_TO_CREATE          = ["OUTPUT_DIR_ALIGNER", "OUTPUT_DIR_STACKER"]