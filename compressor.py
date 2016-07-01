#!/usr/bin/env python

from aligner import Aligner
from stacker import Stacker

import os
import sys
import support

INPUT_DIR_ALIGNER   = "export"
TRANSLATION_DATA    = "translation_data.json"
OUTPUT_DIR_ALIGNER  = "aligned"

NAMING_PREFIX       = "m18"
INPUT_DIR_STACKER   = OUTPUT_DIR_ALIGNER
OUTPUT_DIR_STACKER  = "stack_" + NAMING_PREFIX + "2"
EXTENSION           = ".jpg"

def print_config():
    pass

def path_check_and_expand(path):


    if not (path.startswith("/") or path.startswith("~")):
        path = os.path.join(BASE_DIR, path)

    if path.startswith("~"):
        path = os.path.expanduser(path)

    return path

def create_if_not_existing(path):
    if not os.path.exists(path):
        print("created directoy: {}".format(path))
        os.makedirs(path)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

aligner = Aligner()
stacker = Stacker()
input_images = []

# transform to absolute paths
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

INPUT_DIR_ALIGNER   = path_check_and_expand(INPUT_DIR_ALIGNER)
INPUT_DIR_STACKER   = path_check_and_expand(INPUT_DIR_STACKER)
OUTPUT_DIR_STACKER  = path_check_and_expand(OUTPUT_DIR_STACKER)
OUTPUT_DIR_ALIGNER  = path_check_and_expand(OUTPUT_DIR_ALIGNER)
TRANSLATION_DATA    = path_check_and_expand(TRANSLATION_DATA)

if not os.path.exists(INPUT_DIR):
    print("INPUT_DIR not found: {}".format(support.Color.BOLD + INPUT_DIR + support.Color.END))
    sys.exit(-1)

create_if_not_existing(OUTPUT_DIR_STACKED)
create_if_not_existing(OUTPUT_DIR_ALIGNED)

# get all file names
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:

        if f == ".DS_Store":
            continue

        if not f.lower().endswith(EXTENSION):
            continue

        if os.path.getsize(os.path.join(INPUT_DIR, f)) < 100:
            continue

        input_images.append(f)

# print len(input_images)

# init aligner
aligner.REFERENCE_IMAGE                 = "/Users/volzotan/Desktop/export/DSC03135.jpg"
aligner.INPUT_DIR                       = INPUT_DIR_ALIGNER
aligner.OUTPUT_DIR                      = OUTPUT_DIR_ALIGNER
aligner.EXTENSION                       = EXTENSION
aligner.TRANSLATION_DATA                = TRANSLATION_DATA
aligner.USE_CORRECTED_TRANSLATION_DATA  = True

#aligner.init()
#aligner.step2()

# init stacker
stacker.INPUT_DIRECTORY     = INPUT_DIR_STACKER
stacker.NAMING_PREFIX       = NAMING_PREFIX
stacker.RESULT_DIRECTORY    = OUTPUT_DIR_STACKER
stacker.BASE_DIR            = BASE_DIR
stacker.EXTENSION           = EXTENSION

stacker.print_config()
stacker.run(input_images)