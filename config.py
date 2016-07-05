# aligner
INPUT_DIR_ALIGNER   = "/var/www/timebox/jpegs"  # TODO
TRANSLATION_DATA    = "translation_data.json"
OUTPUT_DIR_ALIGNER  = "/var/www/timebox/aligned"

# stacker
NAMING_PREFIX       = "summaery"
INPUT_DIR_STACKER   = "/var/www/timebox/jpegs"
OUTPUT_DIR_STACKER  = "/var/www/timebox/stack"
EXTENSION           = ".jpg"

# meta
DIRS_TO_CHECK           = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]
DIRS_TO_CREATE          = ["OUTPUT_DIR_ALIGNER", "OUTPUT_DIR_STACKER"]
DIRS_ABORT_IF_MISSING   = ["INPUT_DIR_ALIGNER", "INPUT_DIR_STACKER"]