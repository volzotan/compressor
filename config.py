# aligner
INPUT_DIR_ALIGNER   = "export"
TRANSLATION_DATA    = "translation_data.json"
OUTPUT_DIR_ALIGNER  = "aligned"

# stacker
NAMING_PREFIX       = "m18"
INPUT_DIR_STACKER   = OUTPUT_DIR_ALIGNER
OUTPUT_DIR_STACKER  = "stack_" + NAMING_PREFIX + "2"
EXTENSION           = ".jpg"

# meta
DIRS_TO_CHECK           = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]
DIRS_ABORT_IF_MISSING   = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]
DIRS_TO_CREATE          = ["OUTPUT_DIR_ALIGNER", "OUTPUT_DIR_STACKER"]