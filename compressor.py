#!/usr/bin/env python3

from aligner import Aligner
from stacker import Stacker

import os
import sys
import support
import yaml

import config


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


def abort_if_missing(path):
    if not os.path.exists(path):
        print("dir not found: {}".format(support.Color.BOLD + path + support.Color.END))
        sys.exit(-1)


def get_all_file_names(input_dir):
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:

            if f == ".DS_Store":
                continue

            if not f.lower().endswith(config.EXTENSION):
                continue

            if os.path.getsize(os.path.join(input_dir, f)) < 100:
                continue

            file_list.append(f)

    return file_list


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

aligner = Aligner()
stacker = Stacker(aligner)
input_images_aligner = []
input_images_stacker = []

# transform to absolute paths
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# expand all paths
for directory in config.DIRS_TO_EXPAND:
    variable = getattr(config, directory)
    setattr(config, directory, path_check_and_expand(variable))

for directory in config.DIRS_TO_CREATE:
    variable = getattr(config, directory)
    create_if_not_existing(variable)

for directory in config.DIRS_ABORT_IF_MISSING:
    variable = getattr(config, directory)
    abort_if_missing(variable)

input_images_aligner = get_all_file_names(config.INPUT_DIR_ALIGNER)
input_images_stacker = get_all_file_names(config.INPUT_DIR_STACKER)

# init aligner
aligner.REFERENCE_IMAGE                 = config.REFERENCE_IMAGE
aligner.INPUT_DIR                       = config.INPUT_DIR_ALIGNER
aligner.OUTPUT_DIR                      = config.OUTPUT_DIR_ALIGNER
aligner.EXTENSION                       = config.EXTENSION
aligner.TRANSLATION_DATA                = config.TRANSLATION_DATA
aligner.RESET_MATRIX_EVERY_LOOP         = config.RESET_MATRIX_EVERY_LOOP
aligner.DOWNSIZE                        = config.DOWNSIZE
aligner.JSON_SAVE_INTERVAL              = config.JSON_SAVE_INTERVAL
aligner.SKIP_TRANSLATION                = config.SKIP_TRANSLATION

#aligner.init()
#aligner.step1(input_images_aligner)
#aligner.init()
#aligner.step2()

# init stacker
stacker.NAMING_PREFIX       = config.NAMING_PREFIX
stacker.INPUT_DIRECTORY     = config.INPUT_DIR_STACKER
stacker.RESULT_DIRECTORY    = config.OUTPUT_DIR_STACKER
stacker.FIXED_OUTPUT_NAME   = config.FIXED_OUTPUT_NAME
stacker.BASE_DIR            = BASE_DIR
stacker.EXTENSION           = config.EXTENSION
stacker.PICKLE_NAME         = config.PICKLE_NAME

stacker.ALIGN               = config.ALIGN
stacker.DISPLAY_CURVE       = config.DISPLAY_CURVE
stacker.APPLY_CURVE         = config.APPLY_CURVE

stacker.DISPLAY_PEAKING     = config.DISPLAY_PEAKING
stacker.APPLY_PEAKING       = config.APPLY_PEAKING
stacker.PEAKING_THRESHOLD   = config.PEAKING_THRESHOLD
stacker.PEAKING_MUL_FACTOR  = config.PEAKING_MUL_FACTOR

stacker.WRITE_METADATA      = config.WRITE_METADATA
stacker.SORT_IMAGES         = config.SORT_IMAGES

stacker.SAVE_INTERVAL       = config.SAVE_INTERVAL
stacker.PICKLE_INTERVAL     = config.PICKLE_INTERVAL

stacker.print_config()
stacker.run(input_images_stacker)