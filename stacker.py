from PIL import Image
import cv2
import json
import sys, os
import pickle
import datetime
import subprocess
import numpy as np

import pyexiv2
from fractions import Fraction


"""
    Stacker loads every image in INPUT_DIRECTORY,
    stacks it and writes output to RESULT_DIRECTORY.

    Do not run with pypy! (Saving image is 30-40s slower)

    TODO:
    =====
    compressor should write EXIF Metadata to generated image file
    * compressor version (or Date or git-commit)
    * datetime of first and last image in series
    * total number of images used
    * total exposure time

    and reapply extracted Metadata:
    * focal length
    * GPS Location Data
    * Address Location Data (City, Province-State, Location Code, etc)


    DEPENDENCIES:
    =============

    * openCV 3      reading and writing images
    * gexiv2        writing EXIF data
    -   pyexif2
    *

"""

starttime = datetime.datetime.now()
timer = datetime.datetime.now()

def stop_time(msg=None):
    global timer
    seconds = (datetime.datetime.now() - timer).total_seconds()
    if msg is not None:
        if seconds >= 0.1:
            print(msg.format(seconds, "s"))  
        else: # milliseconds
            print(msg.format(seconds * 1000, "ms"))  

    timer = datetime.datetime.now()

    return seconds

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

PICKLE_NAME         = "stack.pickle"
INPUT_DIRECTORY     = "images"
RESULT_DIRECTORY    = "stack"
DIMENSIONS          = None #(4896, 3264) #(6000, 4000) #(5184, 3136) #(1200, 545)
EXTENSION           = ".tif"

CHANGE_BRIGHTNESS   = False # should brightness_increase be applied?
BRIGHTNESS_INCREASE = 0.80  # the less the brighter: divider * BRIGHTNESS_INCREASE

WRITE_METADATA      = True

SAVE_INTERVAL       = 10
PICKLE_INTERVAL     = -1

#data               = json.load(open("export.json", "rb"))

counter             = 0
processed           = 0

crops               = []
stacked_images      = []

# three dimensional array
# be careful about using types that are big enough to prevent overflows
tresor = None

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def write_pickle(tresor, stacked_images):
    print("dump the pickle...")
    pick = {}
    pick["stacked_images"] = stacked_images
    pick["tresor"]         = tresor

    pickle.dump(pick, open(PICKLE_NAME, "wb"))
    del pick

def save():

    global timer
    global processed

    divider = int(round(counter * BRIGHTNESS_INCREASE, 0)) if CHANGE_BRIGHTNESS else counter

    endtime = datetime.datetime.now()
    timediff = endtime - timer
    timer = datetime.datetime.now()

    filepath = os.path.join(RESULT_DIRECTORY, str(counter) + EXTENSION)

    t = tresor / divider

    # convert to uint16 for saving, 0.5s faster than usage of t.astype(np.uint16)
    s = np.asarray(t, np.uint16)
    cv2.imwrite(filepath, s)

    timeperimage = (timediff/processed).total_seconds() if processed != 0 else 0
    processed    = 0 # reset

    print("saved. counter: {} time total: {} saving image: {} time per image: {}".format(counter, timediff, stop_time(), timeperimage))
    return filepath


def read_metadata(images):
    info = {}

    # exposure time
    earliest_image  = None
    latest_image    = None

    for image in images:
        metadata = pyexiv2.ImageMetadata(os.path.join(INPUT_DIRECTORY, image))
        metadata.read()

        timetaken = metadata["Exif.Photo.DateTimeOriginal"].value

        if earliest_image is None or earliest_image[1] > timetaken:
            earliest_image = (image, timetaken)

        if latest_image is None or latest_image[1] < timetaken:
            latest_image = (image, timetaken)

    if earliest_image is not None and latest_image is not None:
        info["exposure_time"] = (latest_image[1] - earliest_image[1]).total_seconds()
    else: 
        info["exposure_time"] = 0
        print("exposure_time could not be computed")

    # capture date
    if latest_image is not None:
        info["capture_date"] = latest_image[1]
    else:
        info["capture_date"] = datetime.datetime.now()
        print("exposure_time could not be computed")

    # number of images
    info["exposure_count"] = len(images)

    # focal length
    metadata = pyexiv2.ImageMetadata(os.path.join(INPUT_DIRECTORY, images[0]))
    metadata.read()
    try:
        info["focal_length"] = metadata["Exif.Photo.FocalLength"].value
    except:
        print("EXIF: focal length missing")

    # compressor version
    try:
        info["version"] = subprocess.check_output(["git", "describe", "--always"])
        if info["version"][-1] == "\n":
            info["version"] = info["version"][:-1]
    except Exception as e:
        print(str(e))
        info["version"] = "not-available"

    # compressing date
    info["compressing_date"] = datetime.datetime.now()

    return info


def write_metadata(filepath, info):
    metadata = pyexiv2.ImageMetadata(filepath)
    metadata.read()

    compressor_name = "compressor v[{}]".format(info["version"])

    # Exif.Image.ProcessingSoftware is overwritten by Lightroom when the final export is done
    key = "Exif.Image.ProcessingSoftware";  metadata[key] = pyexiv2.ExifTag(key, compressor_name)
    key = "Exif.Image.Software";            metadata[key] = pyexiv2.ExifTag(key, compressor_name)

    key = "Exif.Image.Artist";              metadata[key] = pyexiv2.ExifTag(key, "Christopher Getschmann")
    key = "Exif.Image.Copyright";           metadata[key] = pyexiv2.ExifTag(key, "CreativeCommons BY-NC 4.0")
    key = "Exif.Image.ExposureTime";        metadata[key] = pyexiv2.ExifTag(key, Fraction(info["exposure_time"]))
    key = "Exif.Image.ImageNumber";         metadata[key] = pyexiv2.ExifTag(key, info["exposure_count"])
    key = "Exif.Image.DateTimeOriginal";    metadata[key] = pyexiv2.ExifTag(key, info["capture_date"])
    key = "Exif.Image.DateTime";            metadata[key] = pyexiv2.ExifTag(key, info["compressing_date"])

    # TODO Focal Length
    key = "Exif.Image.FocalLength";         metadata[key] = pyexiv2.ExifTag(key, info["focal_length"])
    # TODO GPS Location

    metadata.write()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# load pickle and init variables
try:
    pick = pickle.load(open(PICKLE_NAME, "rb"))
    stacked_images = pick["stacked_images"]
    tresor = pick["tresor"]
    print("pickle loaded. resume with {} images".format(len(stacked_images)))
except Exception as e:
    print(str(e))

stop_time("pickle loading: {}{}")

# get all file names
for root, dirs, files in os.walk(INPUT_DIRECTORY):
    for f in files:

        if f == ".DS_Store":
            continue

        if not f.endswith(EXTENSION):
            continue

        if os.path.getsize(os.path.join(INPUT_DIRECTORY, f)) < 100:
            continue

        crops.append(f)

LIMIT = len(crops)

if LIMIT <= 0:
    print("no images found. exit.")
    sys.exit(-1)

stop_time("searching for files: {}{}")
print("number of images: {}".format(LIMIT))

if WRITE_METADATA:
    metadata = read_metadata(crops)

if DIMENSIONS is None:
    shape = cv2.imread(os.path.join(INPUT_DIRECTORY, crops[0])).shape
    DIMENSIONS = (shape[1], shape[0])

tresor = np.zeros((DIMENSIONS[1], DIMENSIONS[0], 3), dtype=np.uint64)
stop_time("initialization: {}{}")

for f in crops:

    counter += 1

    if f in stacked_images:
        continue

    # 3: read input as 16bit color TIFF
    im = cv2.imread(os.path.join(INPUT_DIRECTORY, f), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    #data = np.array(im, np.int) # 100ms slower per image
    data = np.asarray(im, np.uint64)
    tresor = np.add(tresor, np.uint64(data))

    stacked_images.append(f)

    processed += 1

    if counter >= LIMIT:
        save()
        if PICKLE_INTERVAL > 0:
            write_pickle(tresor, stacked_images)
        break

    if PICKLE_INTERVAL > 0 and counter % PICKLE_INTERVAL == 0:
        write_pickle(tresor, stacked_images)

    if SAVE_INTERVAL > 0 and counter % SAVE_INTERVAL == 0:
        save()

filepath = save()
if WRITE_METADATA:
    write_metadata(filepath, metadata)
sys.exit(0)