#!/usr/bin/env python

from PIL import Image
import cv2
import json
import sys, os
import pickle
import datetime
import subprocess
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
from fractions import Fraction
import matplotlib.pyplot as plt
import support

import gi
gi.require_version('GExiv2', '0.10')
from gi.repository import GExiv2


"""
    Stacker loads every image in INPUT_DIRECTORY,
    stacks it and writes output to RESULT_DIRECTORY.

    Do not run with pypy! (Saving image is 30-40s slower)


    =====
    compressor writes EXIF Metadata to generated image file
    * compressor version (or Date or git-commit)
    * datetime of first and last image in series
    * total number of images used
    * total exposure time

    and (should, still TODO) reapply extracted Metadata:
    * focal length
    * camera model
    * GPS Location Data
    * Address Location Data (City, Province-State, Location Code, etc)

    TODO:

    argparse
    improve metadata reading

"""

class Stopwatch(object):

    def __init__(self):
        pass

    def stop(self, tag):
        pass


class Stacker(object):

    NAMING_PREFIX       = ""
    INPUT_DIRECTORY     = "images"
    RESULT_DIRECTORY    = "stack_" + NAMING_PREFIX
    FIXED_OUTPUT_NAME   = None
    DIMENSIONS          = None # (length, width)
    EXTENSION           = ".tif"

    BASE_DIR            = None

    PICKLE_NAME         = "stack.pickle"

    # Align
    ALIGN                           = False
    USE_CORRECTED_TRANSLATION_DATA  = False

    # Curve
    DISPLAY_CURVE                   = False
    APPLY_CURVE                     = False

    # Peaking
    APPLY_PEAKING                   = True
    PEAKING_THRESHOLD               = 200 # TODO: don't use fixed values
    PEAKING_MUL_FACTOR              = 1.0
    PEAKING_BLUR                    = True
    PEAKING_GAUSSIAN_FILTER_SIZE    = 1

    WRITE_METADATA      = True
    SORT_IMAGES         = True

    SAVE_INTERVAL       = 15
    PICKLE_INTERVAL     = -1

    DEBUG               = True

    # misc

    EXIF_DATE_FORMAT    = '%Y:%m:%d %H:%M:%S'

    # debug options

    DISPLAY_PEAKING     = False
    CLIPPING_VALUE      = 254

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    def __init__(self, aligner):
        #data               = json.load(open("export.json", "rb"))

        self.aligner                    = aligner

        self.counter                    = 0
        self.processed                  = 0

        self.input_images               = []
        self.stacked_images             = []

        self.weighted_average_divider   = 0

        self.stopwatch           = {
            "load_image": 0,
            "transform_image": 0,
            "convert_to_array": 0,
            "curve": 0,
            "peaking": 0,
            "adding": 0,
            "write_image": 0
        }

        self.tresor = None
        self.peaking_tresor = None

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    def print_config(self):

        config = [
        ("directories:", ""),
        ("   input:  {}", self.INPUT_DIRECTORY),
        ("   output: {}", self.RESULT_DIRECTORY),
        ("   prefix: {}", self.NAMING_PREFIX),
        ("extension: {}", self.EXTENSION),
        (" ", " "),
        ("modifications:", ""),
        ("   align:   {}", str(self.ALIGN)),
        ("   curve:   {}", str(self.APPLY_CURVE)),
        ("   peaking: {}", str(self.APPLY_PEAKING)),
        (" ", " "),
        ("save interval:   {}", str(self.SAVE_INTERVAL)),
        ("pickle interval: {}", str(self.PICKLE_INTERVAL))
        ]

        for line in config:
            if len(line) > 1:
                print(line[0].format(support.Color.BOLD + line[1] + support.Color.END))
            else:
                print(line)

        print("---------------------------------------")


    def write_pickle(self, tresor, stacked_images):
        print("dump the pickle...")
        pick = {}
        pick["stacked_images"] = stacked_images
        pick["tresor"]         = tresor

        pickle.dump(pick, open(self.PICKLE_NAME, "wb"))
        del pick


    def save(self, fixed_name=None):

        filename = None

        if fixed_name is None:
            filename = str(self.counter) + self.EXTENSION
            if len(self.NAMING_PREFIX) > 0:
                filename = self.NAMING_PREFIX + "_" + filename
        else:
            filename = fixed_name

        filepath = os.path.join(self.RESULT_DIRECTORY, filename)

        t = self.tresor.copy()

        overflow_perc = np.amax(t) / (np.iinfo(np.uint64).max / 100.0)
        if overflow_perc > 70:
            print("tresor overflow status: {}%".format(round(overflow_perc, 2)))

        if self.APPLY_CURVE:
            t = t / (self.weighted_average_divider)
        else:
            t = t / (self.counter)



        if self.APPLY_PEAKING:

            """
            different methods of peaking:
            * sum up all the peaking values, apply the multiplication factor and add to tresor before the dividing happens
            * sum up all the peaking values, clip the limits at the max allowed value

            PEAKING_MUL_FACTOR doesn't need to be a fixed value, it may be also something like counter/2

            """

            if self.DEBUG:
                print("saving: max value before peaking in image: {}".format(np.amax(t)))

            # clip to max value
            peaked = np.clip(self.peaking_tresor, 0, self.CLIPPING_VALUE)

            # blur the result to avoid sharp edges
            if self.PEAKING_BLUR:
                peaked = gaussian_filter(peaked, sigma=self.PEAKING_GAUSSIAN_FILTER_SIZE)

            peaked = np.asarray(peaked * self.PEAKING_MUL_FACTOR, np.uint64)

            t = np.add(t, peaked)

            s = np.asarray(peaked, np.uint16)
            cv2.imwrite(filepath + ".peaking.jpg", s)

        if self.DEBUG:
            print("saving: max value in image: {}".format(np.amax(t)))

        # TODO: check for any overflows of single pixels
        # e.g.: through peaking for example some pixels may be brighter than allows.
        #       those need to be capped

        t = np.clip(t, 0, self.CLIPPING_VALUE)

        # convert to uint16 for saving, 0.5s faster than usage of t.astype(np.uint16)
        s = np.asarray(t, np.uint16)
        cv2.imwrite(filepath, s)

        self.stopwatch["write_image"] += self.stop_time()

        # time calculations

        timeperimage = 0
        for key in self.stopwatch:
            timeperimage += self.stopwatch[key]
        timeperimage -= self.stopwatch["write_image"]
        timeperimage /= self.counter

        images_remaining = (self.LIMIT - self.counter) 
        est_remaining = images_remaining * timeperimage + ( (images_remaining/self.SAVE_INTERVAL) * (self.stopwatch["write_image"]/self.counter) )

        save_time = self.stopwatch["write_image"]/(self.counter/self.SAVE_INTERVAL)

        print("saved. counter: {0:3d} time total: {1:3d} saving image: {2:3d} time per image: {3:3d} est. remaining: {4:5d} || {5}".format(self.counter, int((datetime.datetime.now()-self.starttime).total_seconds()), int(save_time), int(timeperimage), int(est_remaining), support.Converter().humanReadableSeconds(est_remaining)))
        #print self.stopwatch
        return filepath


    def read_metadata(self, images):
        info = {}

        # exposure time
        earliest_image  = None
        latest_image    = None

        #gi.require_version('GExiv2', '0.10')
        #from gi.repository import GExiv2

        for image in images:
            metadata = GExiv2.Metadata()

            # for item in dir(metadata):
            #     print(item)

            metadata.open_path(os.path.join(self.INPUT_DIRECTORY, image))

            timetaken = datetime.datetime.strptime(metadata.get_tag_string("Exif.Photo.DateTimeOriginal"), self.EXIF_DATE_FORMAT)

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
        metadata = GExiv2.Metadata()
        metadata.open_path(os.path.join(self.INPUT_DIRECTORY, images[0]))

        info["focal_length"] = metadata.get_focal_length()
        if info["focal_length"] < 0:
            print("EXIF: focal length missing")
            info["focal_length"] = None

        # compressor version
        try:
            info["version"] = subprocess.check_output(["git", "describe", "--always"], cwd=self.BASE_DIR)
            info["version"] = info["version"].decode("utf-8")
            if info["version"][-1] == "\n":
                info["version"] = info["version"][:-1]
        except Exception as e:
            print(str(e))
            info["version"] = "not-available"

        # compressing date
        info["compressing_date"] = datetime.datetime.now()

        return info


    def write_metadata(self, filepath, info):
        metadata = GExiv2.Metadata()
        metadata.open_path(filepath)

        compressor_name = "compressor v[{}]".format(info["version"])
        print(compressor_name)

        # Exif.Image.ProcessingSoftware is overwritten by Lightroom when the final export is done
        key = "Exif.Image.ProcessingSoftware";  metadata.set_tag_string(key, compressor_name)
        key = "Exif.Image.Software";            metadata.set_tag_string(key, compressor_name)

        key = "Exif.Image.Artist";              metadata.set_tag_string(key, "Christopher Getschmann")
        key = "Exif.Image.Copyright";           metadata.set_tag_string(key, "CreativeCommons BY-NC 4.0")
        key = "Exif.Image.ExposureTime";        metadata.set_exif_tag_rational(key, info["exposure_time"], 1)
        key = "Exif.Image.ImageNumber";         metadata.set_tag_long(key, info["exposure_count"])
        key = "Exif.Image.DateTimeOriginal";    metadata.set_tag_string(key, info["capture_date"].strftime(self.EXIF_DATE_FORMAT))
        key = "Exif.Image.DateTime";            metadata.set_tag_string(key, info["compressing_date"].strftime(self.EXIF_DATE_FORMAT))

        if info["focal_length"] is not None:
            key = "Exif.Image.FocalLength";     metadata.set_exif_tag_rational(key, info["focal_length"], 1)
        # TODO GPS Location

        metadata.save_file(filepath)
        print("metadata written to {}".format(filepath))


    def _intensity(self, shutter, aperture, iso):

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


    def calculate_brightness_curve(self, images):
        curve = []

        for image in images:
            metadata = pyexiv2.ImageMetadata(os.path.join(self.INPUT_DIRECTORY, image))
            metadata.read()

            shutter = float(metadata["Exif.Photo.ExposureTime"].value)
            iso     = metadata["Exif.Photo.ISOSpeedRatings"].value

            try: 
                time = metadata["Exif.Image.DateTimeOriginal"].value
            except KeyError as e:
                time = metadata["Exif.Image.DateTime"].value

            try:
                aperture = float(metadata["Exif.Photo.ApertureValue"].value)
            except KeyError as e:
                # no aperture tag set, probably an lens adapter was used. assume fixed aperture.
                aperture = None

            curve.append((image, time, self._intensity(shutter, aperture, iso)))

        # normalize
        values = [x[2] for x in curve]

        min_brightness = min(values)
        max_brightness = max(values)

        for i in range(0, len(curve)):
            # range 0 to 1, because we have to invert the camera values to derive the brightness
            # value of the camera environment

            image_name                          = curve[i][0]
            time                                = curve[i][1]
            relative_brightness_value           = np.interp(curve[i][2], [min_brightness, max_brightness], [1, 0]) # range [0;1]
            inverted_absolute_value             = np.interp(curve[i][2], [min_brightness, max_brightness], [max_brightness, min_brightness])

            # right now the inverted absolute brightness, which is used for the weighted curve calculation,
            # is quite a large number. Usually around 20. (but every image is multiplied with it's respective value,
            # resulting in enourmous numbers in the tresor matrix)
            #
            # better: value - min_brightness + 1 (result should never actually be zero)

            # inverted_absolute_value = inverted_absolute_value - min_brightness + 1

            curve[i] = (image_name, time, relative_brightness_value, inverted_absolute_value)

        values = [x[2] for x in curve]
        self.curve_avg = sum(values) / float(len(values))

        return curve


    def display_curve(self, curve):
        dates = [i[1] for i in curve]
        values = [i[3] for i in curve]

        plt.plot(dates, values)
        plt.show()


    def apply_peaking(self, data):

        datacopy = data.copy()

        # calculate boolean mask for every color channel
        mask_rgb = data > self.PEAKING_THRESHOLD

        # combine mask via AND
        # all three channels must be > PEAKING_THRESHOLD
        mask_all_channels = np.logical_and(mask_rgb[:,:,0], mask_rgb[:,:,1], mask_rgb[:,:,2])

        # invert mask and set everything to 0 if not all three channels are > threshold
        datacopy[~mask_all_channels] = 0

        # TODO: improvement:
        # right now the whole image (data) gets copied, certain parts are nulled and everything will
        # be added to peaking_tresor. Better: just add the non masked parts from the original data ndarray.

        if self.DISPLAY_PEAKING:
           
            #plt.imshow(~mask_all_channels, cmap="Greys", vmin=0, vmax=1)

            s = np.asarray(datacopy, np.uint16)
            cv2.imwrite(os.path.join(self.RESULT_DIRECTORY, "plot.jpg"), s)

            #plt.imshow(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
            self._plot(datacopy)

            # plt.show()
            sys.exit(0)

        self.peaking_tresor = np.add(self.peaking_tresor, datacopy)
        self.stopwatch["peaking"] += self.stop_time()


    def stop_time(self, msg=None):
        seconds = (datetime.datetime.now() - self.timer).total_seconds()
        if msg is not None:
            if seconds >= 0.1:
                print(msg.format(seconds, "s"))  
            else: # milliseconds
                print(msg.format(seconds * 1000, "ms"))  

        self.timer = datetime.datetime.now()

        return seconds


    def reset_timer(self):
        self.timer = datetime.datetime.now()


    def _sort_helper(self, value):

        # still_123.jpg

        pos = value.index(".")
        number = value[6:pos]
        return int(number)


    def _plot(self, mat):
        # plot a numpy array with matplotlib
        plt.imshow(cv2.bitwise_not(cv2.cvtColor(np.asarray(mat, np.uint16), cv2.COLOR_RGB2BGR)), interpolation="nearest")
        plt.show()

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    def run(self, inp_imgs):

        self.input_images = inp_imgs

        self.starttime = datetime.datetime.now()
        self.timer = datetime.datetime.now()

        # load pickle and init variables
        try:
            pick = pickle.load(open(self.PICKLE_NAME, "rb"))
            self.stacked_images = pick["stacked_images"]
            self.tresor = pick["tresor"]
            print("pickle loaded. resume with {} images".format(len(stacked_images)))
        except Exception as e:
            print(str(e))

        self.stop_time("pickle loading: {}{}")

        self.LIMIT = len(self.input_images)
        self.LIMIT = 1

        if self.LIMIT <= 0:
            print("no images found. exit.")
            sys.exit(-1)

        self.stop_time("searching for files: {}{}")
        print("number of images: {}".format(self.LIMIT))

        if self.SORT_IMAGES:
            self.input_images = sorted(self.input_images, key=self._sort_helper)

        if self.WRITE_METADATA:
            self.metadata = self.read_metadata(self.input_images)

        if self.DIMENSIONS is None:
            shape = cv2.imread(os.path.join(self.INPUT_DIRECTORY, self.input_images[0])).shape
            self.DIMENSIONS = (shape[1], shape[0])

        self.tresor = np.zeros((self.DIMENSIONS[1], self.DIMENSIONS[0], 3), dtype=np.uint64)
        if self.APPLY_PEAKING:
            self.peaking_tresor = np.zeros((self.DIMENSIONS[1], self.DIMENSIONS[0], 3), dtype=np.uint64)
        self.stop_time("initialization: {}{}")

        # Curve
        if self.DISPLAY_CURVE or self.APPLY_CURVE:
            self.curve = self.calculate_brightness_curve(self.input_images)
            self.stop_time("compute brightness curve: {}{}")

        if self.DISPLAY_CURVE:
            self.display_curve(self.curve)

        if self.ALIGN:
            self.translation_data = json.load(open(self.aligner.TRANSLATION_DATA, "r"))

        for f in self.input_images:

            self.counter += 1

            if f in self.stacked_images:
                continue

            # read input as 16bit color TIFF
            im = cv2.imread(os.path.join(self.INPUT_DIRECTORY, f), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            self.stopwatch["load_image"] += self.stop_time()

            if self.ALIGN:

                if self.USE_CORRECTED_TRANSLATION_DATA:
                    # translation_data[f] = ( (computed_x, computed_y), (corrected_x, corrected_y) ) 
                    (x, y) = (self.translation_data[f][2][0], self.translation_data[f][2][1])
                else:
                    (x, y) = (self.translation_data[f][1][0], self.translation_data[f][1][1])

                im = self.aligner.transform(im, x, y)
                self.stopwatch["transform_image"] += self.stop_time()

            # data = np.array(im, np.int) # 100ms slower per image
            data = np.uint64(np.asarray(im, np.uint64))
            self.stopwatch["convert_to_array"] += self.stop_time()

            if not self.APPLY_CURVE:
                self.tresor = np.add(self.tresor, data)

            if self.APPLY_CURVE:
                multiplier = self.curve[self.counter-1][3]
                self.tresor = np.add(self.tresor, data * multiplier)
                self.weighted_average_divider += multiplier

            self.stopwatch["adding"] += self.stop_time()

            if self.APPLY_PEAKING:
                self.apply_peaking(data)

            self.stacked_images.append(f)

            if self.counter >= self.LIMIT:
                if self.PICKLE_INTERVAL > 0:
                    self.write_pickle(self.tresor, self.stacked_images)
                break

            if self.PICKLE_INTERVAL > 0 and self.counter % self.PICKLE_INTERVAL == 0:
                self.write_pickle(tresor, stacked_images)
                self.reset_timer()

            if self.SAVE_INTERVAL > 0 and self.counter % self.SAVE_INTERVAL == 0:
                self.save()

            print("counter: {0:.0f}/{1:.0f}".format(self.counter, len(self.input_images)), end="\r")
        

        filepath = self.save(fixed_name=self.FIXED_OUTPUT_NAME)
        if self.WRITE_METADATA:
            self.write_metadata(filepath, self.metadata)

        print("finished. time total: {}".format(datetime.datetime.now() - self.starttime))
        sys.exit(0)
