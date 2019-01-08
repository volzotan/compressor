#!/usr/bin/env python3

from aligner import Aligner
from stacker import Stacker

import argparse

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

    if config.EXTENSION is not None:

        file_list = []

        # recursive:
        # for root, dirs, files in os.walk(input_dir):
        #     for f in files:

        counter = 0

        for f in os.listdir(input_dir):

            if f == ".DS_Store":
                counter += 1
                continue

            if not f.lower().endswith(config.EXTENSION):
                counter += 1
                continue

            if os.path.getsize(os.path.join(input_dir, f)) < 100:
                counter += 1
                continue

            file_list.append(f)

        if counter > 0:
            print("skipped {} files during parsing of directory {}".format(counter, input_dir))

        return file_list

    else: # EXTENSION autodetect

        file_list_jpg = []
        file_list_tif = []

        for f in os.listdir(input_dir):
   
            if f.lower().endswith(".jpg"):
                file_list_jpg.append(f)

            if f.lower().endswith(".tif"):
                file_list_tif.append(f)

        print("Extension autodetection: {} JPGs, {} TIFs found.".format(len(file_list_jpg), len(file_list_tif)))

        if (len(file_list_jpg) > 0 and len(file_list_jpg) >= len(file_list_tif)):
            print("Extension autodetection: JPG choosen")
            config.EXTENSION = ".jpg"
            return file_list_jpg

        if (len(file_list_tif) > 0 and len(file_list_tif) >= len(file_list_jpg)):
            print("Extension autodetection: TIF choosen")
            config.EXTENSION = ".tif"
            return file_list_tif

        print("Extension autodetection failed")
    
    return []


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

parser = argparse.ArgumentParser(description="Stack several image files to create digital long exposure photographies")
parser.add_argument("--align", action="store_true", help="run only the aligner, do not compress")
args = parser.parse_args()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

aligner = Aligner()
stacker = Stacker(aligner)
input_images_aligner = []
input_images_stacker = []

# transform to absolute paths
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# init aligner

if args.align:

    # expand all paths
    for directory in config.DIRS_TO_EXPAND_ALIGNER:
        variable = getattr(config, directory)
        if variable is None:
            print("warning: variable {} empty. not expanded".format(directory))
            continue
        setattr(config, directory, path_check_and_expand(variable))

    for directory in config.DIRS_TO_CREATE_ALIGNER:
        variable = getattr(config, directory)
        create_if_not_existing(variable)

    for directory in config.DIRS_ABORT_IF_MISSING_ALIGNER:
        variable = getattr(config, directory)
        abort_if_missing(variable)

    input_images_aligner = get_all_file_names(config.INPUT_DIR_ALIGNER)

    if config.REFERENCE_IMAGE is None:
        if len(input_images_aligner) == 0:
            print("aligner: no input images. abort.")
            sys.exit(-1)
        config.REFERENCE_IMAGE = os.path.join(path_check_and_expand(config.INPUT_DIR_ALIGNER), input_images_aligner[0])
        print(config.REFERENCE_IMAGE)
        print("aligner: no reference image specified. defaulting to first image.")

    aligner.REFERENCE_IMAGE                 = config.REFERENCE_IMAGE
    aligner.INPUT_DIR                       = config.INPUT_DIR_ALIGNER
    aligner.OUTPUT_DIR                      = config.OUTPUT_DIR_ALIGNER
    aligner.EXTENSION                       = config.EXTENSION
    aligner.TRANSLATION_DATA                = config.TRANSLATION_DATA
    aligner.RESET_MATRIX_EVERY_LOOP         = config.RESET_MATRIX_EVERY_LOOP
    aligner.DOWNSIZE                        = config.DOWNSIZE
    aligner.DOWNSIZE_FACTOR                 = config.DOWNSIZE_FACTOR
    aligner.JSON_SAVE_INTERVAL              = config.JSON_SAVE_INTERVAL
    aligner.SKIP_TRANSLATION                = config.SKIP_TRANSLATION

    aligner.init()
    aligner.step1(input_images_aligner)
    # aligner.init()
    # aligner.step2()


# init stacker

if not args.align:

    # expand all paths
    for directory in config.DIRS_TO_EXPAND_STACKER:
        variable = getattr(config, directory)
        if variable is None:
            print("warning: variable {} empty. not expanded".format(directory))
            continue
        setattr(config, directory, path_check_and_expand(variable))

    for directory in config.DIRS_TO_CREATE_STACKER:
        variable = getattr(config, directory)
        create_if_not_existing(variable)

    for directory in config.DIRS_ABORT_IF_MISSING_STACKER:
        variable = getattr(config, directory)
        abort_if_missing(variable)

    input_images_stacker = get_all_file_names(config.INPUT_DIR_STACKER)

    stacker.NAMING_PREFIX       = config.NAMING_PREFIX
    stacker.INPUT_DIRECTORY     = config.INPUT_DIR_STACKER
    stacker.RESULT_DIRECTORY    = config.OUTPUT_DIR_STACKER

    if config.FIXED_OUTPUT_NAME.endswith(config.EXTENSION):
        stacker.FIXED_OUTPUT_NAME   = config.FIXED_OUTPUT_NAME
    else:
        stacker.FIXED_OUTPUT_NAME   = config.FIXED_OUTPUT_NAME + config.EXTENSION

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
    stacker.INTERMEDIATE_SAVE_FORCE_JPEG = config.INTERMEDIATE_SAVE_FORCE_JPEG
    stacker.PICKLE_INTERVAL     = config.PICKLE_INTERVAL

    stacker.post_init()
    stacker.print_config()
    stacker.run(input_images_stacker)
