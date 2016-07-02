#!/usr/bin/env python

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

            if not f.lower().endswith(EXTENSION):
                continue

            if os.path.getsize(os.path.join(INPUT_DIR, f)) < 100:
                continue

            file_list.append(f)


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

aligner = Aligner()
stacker = Stacker()
input_images_aligner = []
input_images_stacker = []

# transform to absolute paths
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# expand all paths
for directory in config.DIRS_TO_CHECK:
    variable = getattr(config, directory)
    setattr(config, directory, path_check_and_expand(variable))

for directory in config.DIRS_ABORT_IF_MISSING:
    abort_if_missing(directory)

for directory in config.DIRS_TO_CREATE:
    create_if_not_existing(directory)

input_images_aligner = get_all_file_names(config.INPUT_DIR_ALIGNER)
input_images_stacker = get_all_file_names(config.INPUT_DIR_STACKER)

# print len(input_images)

# init aligner
aligner.REFERENCE_IMAGE                 = "/Users/volzotan/Desktop/export/DSC03135.jpg"
aligner.INPUT_DIR                       = config.INPUT_DIR_ALIGNER
aligner.OUTPUT_DIR                      = config.OUTPUT_DIR_ALIGNER
aligner.EXTENSION                       = config.EXTENSION
aligner.TRANSLATION_DATA                = config.TRANSLATION_DATA
aligner.USE_CORRECTED_TRANSLATION_DATA  = True

#aligner.init()
#aligner.step2()

# init stacker
stacker.INPUT_DIRECTORY     = config.INPUT_DIR_STACKER
stacker.NAMING_PREFIX       = config.NAMING_PREFIX
stacker.RESULT_DIRECTORY    = config.OUTPUT_DIR_STACKER
stacker.BASE_DIR            = BASE_DIR
stacker.EXTENSION           = config.EXTENSION

stacker.print_config()
stacker.run(input_images)