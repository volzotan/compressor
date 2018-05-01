# aligner
INPUT_DIR_ALIGNER       = "~/Desktop/export_TIFF"
TRANSLATION_DATA        = "translation_data.json"
OUTPUT_DIR_ALIGNER      = "aligned"

REFERENCE_IMAGE         = "" #"~/Desktop/peakingtest/DSC04370.jpg"

RESET_MATRIX_EVERY_LOOP = False
DOWNSIZE                = True
SKIP_TRANSLATION        = -1

JSON_SAVE_INTERVAL      = 10

# stacker
NAMING_PREFIX           = ""
INPUT_DIR_STACKER       = "/Users/volzotan/Documents/DESPATDATASETS/18-04-09_darmstadt_motoZ"
OUTPUT_DIR_STACKER      = INPUT_DIR_STACKER + "_output"
EXTENSION               = ".jpg"
FIXED_OUTPUT_NAME       = "dmsdt_" + EXTENSION
PICKLE_NAME             = "stack.pickle"

ALIGN                   = False

APPLY_CURVE             = True

APPLY_PEAKING           = False
PEAKING_THRESHOLD       = -1    # auto
PEAKING_MUL_FACTOR      = 0.4

WRITE_METADATA          = True            
SORT_IMAGES             = False

SAVE_INTERVAL           = 50
INTERMEDIATE_SAVE_FORCE_JPEG  = False # still buggy (values to high for jpeg saving)
PICKLE_INTERVAL         = -1

# debug

DISPLAY_CURVE           = False
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