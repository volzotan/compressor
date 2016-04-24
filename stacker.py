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
import math


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

input_images        = []
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
        info["focal_length"] = None

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


def _intensity(shutter, aperture, iso):

    # limits in this calculations:
    # min shutter is 1/4000th second
    # min aperture is 22
    # min iso is 100

    shutter_repr    = math.log(shutter, 2) + 13 # offset = 13 to accomodate shutter values down to 1/4000th second
    iso_repr        = math.log(iso/100, 2) + 1  # offset = 1, iso 100 -> 1, not 0

    if aperture is not None:
        aperture_repr = np.interp(math.log(aperture, 2), [0, 4.5], [10, 1])
    else:
        aperture_repr = 1

    return shutter_repr + aperture_repr + iso_repr

def calculate_brightness_curve(images):
    curve = []

    for image in images:
        metadata = pyexiv2.ImageMetadata(os.path.join(INPUT_DIRECTORY, image))
        metadata.read()

        shutter = float(metadata["Exif.Photo.ExposureTime"].value)
        iso     = metadata["Exif.Photo.ISOSpeedRatings"].value
        try:
            aperture = float(metadata["Exif.Photo.ApertureValue"].value)
        except KeyError as e:
            # no aperture tag set, probably an lens adapter was used. assume fixed aperture.
            aperture = None

        curve.append((image, _intensity(shutter, aperture, iso)))

    # normalize
    values = [x[1] for x in curve]

    min_brightness = min(values)
    max_brightness = max(values)

    for i in range(0, len(curve)):
        # range 0 to 1, because we have to invert the camera values to derive the brightness
        # value of the camera environment

        curve[i] = (curve[i][0], np.interp(curve[i][1], [min_brightness, max_brightness], [1, 0]))

    # print curve


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

        if not f.lower().endswith(EXTENSION):
            continue

        if os.path.getsize(os.path.join(INPUT_DIRECTORY, f)) < 100:
            continue

        input_images.append(f)

LIMIT = len(input_images)

if LIMIT <= 0:
    print("no images found. exit.")
    sys.exit(-1)

stop_time("searching for files: {}{}")
print("number of images: {}".format(LIMIT))

if WRITE_METADATA:
    metadata = read_metadata(input_images)

if DIMENSIONS is None:
    shape = cv2.imread(os.path.join(INPUT_DIRECTORY, input_images[0])).shape
    DIMENSIONS = (shape[1], shape[0])

tresor = np.zeros((DIMENSIONS[1], DIMENSIONS[0], 3), dtype=np.uint64)
stop_time("initialization: {}{}")

brightness_index = calculate_brightness_curve(input_images)
stop_time("compute brightness curve: {}{}")

sys.exit(0)

for f in input_images:

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