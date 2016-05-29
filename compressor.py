#!/usr/bin/env python

from aligner import Aligner
from stacker import Stacker

import os
import sys
import support

INPUT_DIR           = "images_hbf1"
OUTPUT_DIR_STACKED  = "foo"
EXTENSION           = ".tif"


def print_config():
    pass


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

#aligner = Aligner()
stacker = Stacker()
input_images = []

# transform to absolute paths
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

if not (INPUT_DIR.startswith("/") or INPUT_DIR.startswith("~")):
    INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR)

if not (OUTPUT_DIR_STACKED.startswith("/") or OUTPUT_DIR_STACKED.startswith("~")):
    OUTPUT_DIR_STACKED = os.path.join(BASE_DIR, OUTPUT_DIR_STACKED)

if not os.path.exists(INPUT_DIR):
    print("INPUT_DIR not found: {}".format(support.Color.BOLD + INPUT_DIR + support.Color.END))
    sys.exit(-1)

# check or create OUTPUT_DIR_STACKED
if not os.path.exists(OUTPUT_DIR_STACKED):
    print("created OUTPUT_DIR_STACKED: {}".format(OUTPUT_DIR_STACKED))
    os.makedirs(OUTPUT_DIR_STACKED)

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

# init stacker
stacker.INPUT_DIRECTORY = INPUT_DIR
stacker.RESULT_DIRECTORY = OUTPUT_DIR_STACKED
stacker.BASE_DIR = BASE_DIR

stacker.print_config()
stacker.run(input_images)