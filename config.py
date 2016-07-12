# aligner
INPUT_DIR_ALIGNER   = "/var/www/timebox/jpegs"  # TODO
TRANSLATION_DATA    = "translation_data.json"
OUTPUT_DIR_ALIGNER  = "/var/www/timebox/aligned"

# stacker
NAMING_PREFIX       = ""
INPUT_DIR_STACKER   = "/var/www/timebox/jpegs"
OUTPUT_DIR_STACKER  = "/var/www/timebox/stack"
FIXED_OUTPUT_NAME   = "summaery.jpg"
EXTENSION           = ".jpg"
PICKLE_NAME         = "stack.pickle"

DISPLAY_CURVE       = False
APPLY_CURVE         = False

DISPLAY_PEAKING     = False
APPLY_PEAKING       = False
PEAKING_THRESHOLD   = 1
PEAKING_MUL_FACTOR  = 1.0

WRITE_METADATA      = False
SORT_IMAGES         = False

SAVE_INTERVAL       = -1
PICKLE_INTERVAL     = 5

# meta
DIRS_TO_CHECK           = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]
DIRS_TO_CREATE          = ["OUTPUT_DIR_ALIGNER", "OUTPUT_DIR_STACKER"]
DIRS_ABORT_IF_MISSING   = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]