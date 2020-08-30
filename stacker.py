#!/usr/bin/env python

import support

import json
import sys
import os
import pickle
import datetime
import subprocess
import logging
import math
from fractions import Fraction

from PIL import Image
# import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import color
import matplotlib.pyplot as plt

import exifread

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
    * camera model
    * GPS Location Data
    * Address Location Data (City, Province-State, Location Code, etc)

    TODO:

    improve metadata reading

"""

BLEND_MODE_STACK                = "stack"
BLEND_MODE_PEAK                 = "peak"

class Stopwatch(object):

    def __init__(self):
        pass

    def stop(self, tag):
        pass


class Stack(object):

    NAMING_PREFIX                   = ""
    INPUT_DIRECTORY                 = "images"
    RESULT_DIRECTORY                = "stack_" + NAMING_PREFIX
    FIXED_OUTPUT_NAME               = None
    DIMENSIONS                      = None # (length, width)
    EXTENSION                       = ".tif"

    BASE_DIR                        = None

    PICKLE_NAME                     = "stack.pickle"

    # Image blend mode (stack / peak)
    BLEND_MODE                      = BLEND_MODE_STACK

    # Align
    ALIGN                           = False
    USE_CORRECTED_TRANSLATION_DATA  = False

    MIN_BRIGHTNESS_THRESHOLD        = 100

    # Curve
    DISPLAY_CURVE                   = False
    APPLY_CURVE                     = False

    WRITE_METADATA                  = True
    SAVE_INTERVAL                   = 15
    PICKLE_INTERVAL                 = -1

    DEBUG                           = False

    # misc

    EXIF_DATE_FORMAT                = '%Y:%m:%d %H:%M:%S'

    # debug options

    CLIPPING_VALUE                  = -1

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    def __init__(self, aligner):

        self.aligner                            = aligner

        self.counter                            = 0
        self.processed                          = 0

        self.input_images                       = []
        self.stacked_images                     = []

        self.weighted_average_divider           = 0

        self.stopwatch                          = {
            "load_image": 0,
            "transform_image": 0,
            "convert_to_array": 0,
            "stacking": 0,
            "write_image": 0
        }

        self.tresor                             = None

        # initialize log

        self.log = logging.getLogger("stacker")
        self.log.setLevel(logging.DEBUG)
        self.log.debug("init")


    """
    After the class variables have been overwritten with the config from the compressor.py script,
    some default values may need to be calculated.
    """
    def post_init(self):

        if self.CLIPPING_VALUE < 0 and self.EXTENSION == ".jpg":
            self.CLIPPING_VALUE = 2**8 - 1
        if self.CLIPPING_VALUE < 0 and self.EXTENSION == ".tif":
            self.CLIPPING_VALUE = 2**16 - 1

        # TODO: is ignored right now
        # self.PEAKING_THRESHOLD = int(self.CLIPPING_VALUE * self.PEAKING_PIXEL_THRESHOLD) # peaking includes pixel above 95% brightness

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    def print_config(self):

        config = [
            ("blend mode:       {}", str(self.BLEND_MODE)),
            (" ", " "),
            ("directories:", ""),
            ("   input:         {}", self.INPUT_DIRECTORY),
            ("   output:        {}", self.RESULT_DIRECTORY),
            ("   prefix:        {}", self.NAMING_PREFIX),
            ("extension:        {}", self.EXTENSION),
            (" ", " "),
            ("modifications:", ""),
            ("   align:         {}", str(self.ALIGN)),
            ("   curve:         {}", str(self.APPLY_CURVE)),
            (" ", " "),
            ("save interval:    {}", str(self.SAVE_INTERVAL)),
            ("pickle interval:  {}", str(self.PICKLE_INTERVAL))
        ]

        self.log.debug("CONFIG:")
        for line in config:
            if len(line) > 1:
                self.log.debug(line[0].format(support.Color.BOLD + line[1] + support.Color.END))
            else:
                self.log.debug(line)

        self.log.debug("---------------------------------------")


    def write_pickle(self, tresor, stacked_images):
        self.log.debug("dump the pickle...")
        pick = {}
        pick["stacked_images"] = stacked_images
        pick["tresor"]         = tresor

        pickle.dump(pick, open(self.PICKLE_NAME, "wb"))
        del pick


    def save(self, fixed_name=None, force_jpeg=False):

        filename = None

        if fixed_name is None:
            # {0:0Nd}{1} where N is used to pad to the length of the highest number of images (eg. 001 for 256 images)
            pattern = str("{0:0") + str(len(str(len(self.input_images)))) + str("d}{1}")
            filename = pattern.format(self.counter, self.EXTENSION)

            if force_jpeg:
                filename = str(self.counter) + ".jpg"

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
            if self.weighted_average_divider > 0:
                t = t / (self.weighted_average_divider)
        else:
            if self.counter > 0:
                t = t / (self.counter)

        # convert to uint16 for saving, 0.5s faster than usage of t.astype(np.uint16)
        s = np.asarray(t, np.uint16)

        # cv2.imwrite(filepath, s)
        Image.fromarray(s).save(filepath)

        if self.WRITE_METADATA:
            self.write_metadata(filepath, self.metadata)

        return filepath


    def read_metadata(self, images):
        info = {}

        # exposure time
        earliest_image  = None
        latest_image    = None

        for image in images:
   
            with open(os.path.join(self.INPUT_DIRECTORY, image), 'rb') as f:
                metadata = exifread.process_file(f)

                try:
                    timetaken = datetime.datetime.strptime(metadata["EXIF DateTimeOriginal"], self.EXIF_DATE_FORMAT)
                except Exception as e:
                    continue;

                if earliest_image is None or earliest_image[1] > timetaken:
                    earliest_image = (image, timetaken)

                if latest_image is None or latest_image[1] < timetaken:
                    latest_image = (image, timetaken)

        if earliest_image is not None and latest_image is not None:
            info["exposure_time"] = (latest_image[1] - earliest_image[1]).total_seconds()
        else: 
            info["exposure_time"] = 0
            print("exposure_time could not be computed")

        with open(os.path.join(self.INPUT_DIRECTORY, images[0]), 'rb') as f:
            metadata = exifread.process_file(f)

            # capture date
            if latest_image is not None:
                info["capture_date"] = latest_image[1]
            else:
                info["capture_date"] = datetime.datetime.now()
                print("exposure_time could not be computed")

            # number of images
            info["exposure_count"] = len(images)

            # focal length
            value = metadata["EXIF FocalLength"].values[0]
            info["focal_length"] = value.num / value.den
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


    def write_metadata_gexiv(self, filepath, info):
    # def write_metadata(self, filepath, info):

        metadata = GExiv2.Metadata()
        metadata.open_path(filepath)

        compressor_name = "compressor v[{}]".format(info["version"])

        # Exif.Image.ProcessingSoftware is overwritten by Lightroom when the final export is done
        key = "Exif.Image.ProcessingSoftware";  metadata.set_tag_string(key, compressor_name)
        key = "Exif.Image.Software";            metadata.set_tag_string(key, compressor_name)

        key = "Exif.Image.Artist";              metadata.set_tag_string(key, "Christopher Getschmann")
        key = "Exif.Image.Copyright";           metadata.set_tag_string(key, "CreativeCommons BY-NC 4.0")
        try:
            key = "Exif.Image.ExposureTime";    metadata.set_exif_tag_rational(key, info["exposure_time"], 1)
        except Exception as e:
            key = "Exif.Image.ExposureTime";    metadata.set_exif_tag_rational(key, info["exposure_time"])
        key = "Exif.Image.ImageNumber";         metadata.set_tag_long(key, info["exposure_count"])
        key = "Exif.Image.DateTimeOriginal";    metadata.set_tag_string(key, info["capture_date"].strftime(self.EXIF_DATE_FORMAT))
        key = "Exif.Image.DateTime";            metadata.set_tag_string(key, info["compressing_date"].strftime(self.EXIF_DATE_FORMAT))

        if info["focal_length"] is not None:
            try:
                key = "Exif.Image.FocalLength"; metadata.set_exif_tag_rational(key, info["focal_length"])
            except Exception as e:
                key = "Exif.Image.FocalLength"; metadata.set_exif_tag_rational(key, info["focal_length"], 1)
        # TODO GPS Location

        metadata.save_file(filepath)
        # print("metadata written to {}".format(filepath))


    def write_metadata(self, filepath, info):
        raise Exception()

    #     metadata = GExiv2.Metadata()
    #     metadata.open_path(filepath)

    #     exif_dict = piexif.load(path)
    #     new_exif = adjust_exif(exif_dict)
    #     exif_bytes = piexif.dump(new_exif)
    #     piexif.insert(exif_bytes, path)



    def _intensity(self, shutter, aperture, iso):

        # limits in this calculations:
        # min shutter is 1/4000th second

        # min aperture is 22
        # apertures: 22  16  11   8 5.6   4 2.8 2.0 1.4   1
        #            10   9   8   7   6   5   4   3   2   1  
        # log-value: 4.4               ...                0

        # min iso is 100

        shutter_repr    = math.log(shutter, 2) + 13 # offset = 13 to accomodate shutter values down to 1/4000th second
        iso_repr        = math.log(iso/100.0, 2) + 1  # offset = 1, iso 100 -> 1, not 0

        if aperture is not None:
            aperture_repr = np.interp(math.log(aperture, 2), [0, 4.5], [10, 1])
        else:
            aperture_repr = 1

        return shutter_repr + aperture_repr + iso_repr


    def _luminosity(self, image):
        return 0

        # TODO: check for broken image?
        #im = cv2.imread(os.path.join(self.INPUT_DIRECTORY, f), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #luma = 0.2126 * R + 0.7152 * G + 0.0722 * B


    def calculate_brightness_curve(self, images):
        values = []

        for image in images:
            metadata = GExiv2.Metadata()
            metadata.open_path(os.path.join(self.INPUT_DIRECTORY, image))

            shutter = metadata.get_exposure_time()
            if shutter[0] == 0 and shutter[1] == 0:
                print("EXIF data missing for image: {}".format(image))
                sys.exit(-1)
            try:
                shutter = float(shutter)
            except TypeError as e:
                if (shutter[1] != 0):
                    shutter = shutter[0] / shutter[1]
                else:
                    shutter = shutter[0]

            iso = metadata.get_tag_string("Exif.Photo.ISOSpeedRatings");
            if iso is not None:
                iso = int(iso)
            else:
                iso = 100 

            time = metadata.get_tag_string("Exif.Photo.DateTimeOriginal")
            if time is None:
                time = metadata.get_tag_string("Exif.Image.DateTime")
                if time is None:
                    # TODO: evil hack
                    # if image.endswith("_0" + self.EXTENSION):
                    #     time = datetime.datetime.fromtimestamp(int(image[:-6])/1000).strftime(self.EXIF_DATE_FORMAT)
                    raise Exception("time exif data missing. no brightness curve can be calculated (well it could, but time data is required for the graph")    
            time = datetime.datetime.strptime(time, self.EXIF_DATE_FORMAT)

            aperture = metadata.get_focal_length()
            if aperture < 0:
                # no aperture tag set, probably an lens adapter was used. assume fixed aperture.
                aperture = None

            values.append((image, time, self._intensity(shutter, aperture, iso), self._luminosity(image)))

        # normalize
        intensities = [x[2] for x in values]

        min_intensity = min(intensities)
        max_intensity = max(intensities)

        curve = []

        for i in range(0, len(values)):
            # range 0 to 1, because we have to invert the camera values to derive the brightness
            # value of the camera environment

            image_name                  = values[i][0]
            time                        = values[i][1]
            relative_brightness_value   = np.interp(values[i][2], [min_intensity, max_intensity], [1, 0]) # range [0;1]
            inverted_absolute_value     = np.interp(values[i][2], [min_intensity, max_intensity], [max_intensity, min_intensity])
            luminosity_value            = values[i][3]

            # right now the inverted absolute brightness, which is used for the weighted curve calculation,
            # is quite a large number. Usually around 20. (but every image is multiplied with it's respective value,
            # resulting in enormous numbers in the tresor matrix)
            #
            # better: value - min_brightness + 1 (result should never actually be zero)

            # inverted_absolute_value = inverted_absolute_value - min_brightness + 1

            # print(inverted_absolute_value)

            curve.append({
                "image_name": image_name, 
                "time": time, 
                "brightness": values[i][2],                         # measure how much light the camera needed to block (lower means scene was brighter)
                "relative_brightness": relative_brightness_value,   # relative inverted brightness (0: darkest scene, 1: brightest scene)
                "inverted_absolute": 2**inverted_absolute_value,    # absolute inverted brightness (min: darkest scene, max: brightest scene)
                "luminosity": luminosity_value                      # 
            })

        relative_brightnesses = [x["relative_brightness"] for x in curve]
        self.curve_avg = sum(relative_brightnesses) / float(len(relative_brightnesses))

        return curve


    def display_curve(self, curve):
        dates = [i["time"] for i in curve]
        values_exif = [i["inverted_absolute"] for i in curve]
        values_luminosity = [i["luminosity"] for i in curve]

        # print(values_exif)

        plt.plot(dates, values_exif)
        plt.plot(dates, values_luminosity)
        # plt.plot(dates, values_luminosity)
        plt.savefig(os.path.join(self.RESULT_DIRECTORY, "curveplot.png"))
        # plt.show()


    def _load_image(self, filename, directory=None):
        # read input as 16bit color TIFF or plain JPG
        if directory is not None:
            image_path = os.path.join(directory, filename)
        else:
            image_path = filename

        # no faster method found for TIF images (tested: PIL, imageio, libtiff) 
        # return cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
        return np.asarray(Image.open(image_path))


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


    # def _plot(self, mat):
    #     # plot a numpy array with matplotlib
    #     plt.imshow(cv2.bitwise_not(cv2.cvtColor(np.asarray(mat, np.uint16), cv2.COLOR_RGB2BGR)), interpolation="nearest")
    #     plt.show()


    def print_info(self):

        # time calculations

        timeperimage = 0
        for key in self.stopwatch:
            timeperimage += self.stopwatch[key]
        timeperimage -= self.stopwatch["write_image"]
        timeperimage /= self.counter

        images_remaining = (self.LIMIT - self.counter) 
        est_remaining = images_remaining * timeperimage + ( (images_remaining/self.SAVE_INTERVAL) * (self.stopwatch["write_image"]/self.counter) )

        save_time = self.stopwatch["write_image"]/(self.counter/self.SAVE_INTERVAL)

        status =  "saved. counter: {:3d} | ".format(         self.counter) 
        status += "time total: {:.1f} | ".format(            (datetime.datetime.now()-self.starttime).total_seconds())
        status += "saving image: {:.1f} | ".format(          save_time) 
        status += "time per image: {:.1f} | ".format(        timeperimage)
        status += "est. remaining: {:.1f} || ".format(       est_remaining)
        status += support.Converter().humanReadableSeconds(  est_remaining)
        
        text =  "load_image: {load_image:.3f} | "
        text += "transform_image: {transform_image:.3f} | "
        text += "convert_to_array: {convert_to_array:.3f} | "
        text += "stacking: {stacking:.3f} | "
        text += "write_image: {write_image:.3f}"

        stopwatch_avg = self.stopwatch.copy()
        for key in stopwatch_avg:
            stopwatch_avg[key] /= self.counter
        # stopwatch_avg["write_image"] *= self.counter
        # stopwatch_avg["write_image"] /= int(self.counter/self.SAVE_INTERVAL)

        # print(status + "\n" + text.format(**stopwatch_avg), end="\r")

        self.log.info("processing image {:5d} / {:5d} ({:5.2f}%) | est. remaining: {}".format(
            self.counter, 
            len(self.input_images), 
            (self.counter/len(self.input_images)) * 100,
            support.Converter().humanReadableSeconds(est_remaining)
        ))

        # print(self.stopwatch)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    def process(self, f):

        self.counter += 1
        if f in self.stacked_images:
            return

        im = self._load_image(f, directory=self.INPUT_DIRECTORY)
        self.stopwatch["load_image"] += self.stop_time()

        skip_processing = False

        # check if the image is too dark to care
        if self.BLEND_MODE == BLEND_MODE_STACK and self.MIN_BRIGHTNESS_THRESHOLD is not None:
                
            # value = np.max(im) # brightest pixel
            value = im.mean() # average brightness

            if value < self.MIN_BRIGHTNESS_THRESHOLD:
                self.log.debug("skipping image: {} (brightness below threshold)".format(f))
                skip_processing = True

        if not skip_processing:

            if self.ALIGN:
                # translation_data[f] = ( matrix, (computed_x, computed_y), (corrected_x, corrected_y) ) 

                warp_matrix_key = None
                for key in self.translation_data.keys():
                    if key.lower() <= f.lower():
                        warp_matrix_key = key
                    else:
                        break

                # if f not in self.translation_data:
                if warp_matrix_key is None:
                    print("not aligned: translation data missing for {}".format(f))
                else:

                    matrix = np.matrix(self.translation_data[warp_matrix_key][0])

                    sum_abs_trans = abs(matrix[0, 2]) + abs(matrix[1, 2])
                    if sum_abs_trans > 100:
                        print("\n   image {} not aligned: values too big: {:.2f}".format(f, sum_abs_trans))
                    else:
                        im = self.aligner.transform(im, matrix, im.shape)
                        self.stopwatch["transform_image"] += self.stop_time()

            # data = np.array(im, np.int) # 100ms slower per image
            data = np.uint64(np.asarray(im, np.uint64))
            self.stopwatch["convert_to_array"] += self.stop_time()

            if self.BLEND_MODE == BLEND_MODE_STACK:

                if self.APPLY_CURVE:
                    multiplier = self.curve[self.counter-1]["inverted_absolute"]
                    data = data * multiplier
                    self.weighted_average_divider += multiplier

                self.tresor = np.add(self.tresor, data)

            elif self.BLEND_MODE == BLEND_MODE_PEAK:

                brighter_mask = self.tresor < data 
                min_brightness_mask = data > 10 #120
                min_brightness_mask = np.any(min_brightness_mask, axis=2, keepdims=True)
                brighter_mask = np.logical_and(brighter_mask, min_brightness_mask)
                self.tresor[brighter_mask] = data[brighter_mask]

            else:   
                self.log.error("unknown BLEND_MODE: {}".format(self.BLEND_MODE))
                exit(-1)

            self.stopwatch["stacking"] += self.stop_time()

        self.stacked_images.append(f)

        # TODO: limit currently disabled because this piece of code is not running directly in a loop anymore
        # if self.counter >= self.LIMIT:
        #     if self.PICKLE_INTERVAL > 0:
        #         self.write_pickle(self.tresor, self.stacked_images)
        #     break

        if self.PICKLE_INTERVAL > 0 and self.counter % self.PICKLE_INTERVAL == 0:
            self.write_pickle(tresor, stacked_images)
            self.reset_timer()

        if self.SAVE_INTERVAL > 0 and self.counter % self.SAVE_INTERVAL == 0:
            self.save(force_jpeg=self.INTERMEDIATE_SAVE_FORCE_JPEG)

        self.stopwatch["write_image"] += self.stop_time()

        self.print_info()

        # print("counter: {0:.0f}/{1:.0f}".format(self.counter, len(self.input_images)), end="\r")


    def run(self, inp_imgs):

        self.input_images = inp_imgs

        # self.input_images = self.input_images[:720]

        self.starttime = datetime.datetime.now()
        self.timer = datetime.datetime.now()

        # load pickle and init variables
        try:
            pick = pickle.load(open(self.PICKLE_NAME, "rb"))
            self.stacked_images = pick["stacked_images"]
            self.tresor = pick["tresor"]
            self.log.info("pickle loaded. resume with {} images".format(len(stacked_images)))
        except Exception as e:
            self.log.error(str(e))

        self.stop_time("pickle loading: {0:.3f}{1}")

        self.LIMIT = len(self.input_images)

        if self.LIMIT <= 0:
            self.log.error("no images found. exit.")
            sys.exit(-1)

        self.stop_time("searching for files: {0:.3f}{1}")
        self.log.info("number of images: {}".format(self.LIMIT))

        if self.WRITE_METADATA:
            self.metadata = self.read_metadata(self.input_images)

        if self.DIMENSIONS is None:
            # cv2.imread(os.path.join(self.INPUT_DIRECTORY, self.input_images[0])).shape
            shape = self._load_image(self.input_images[0], directory=self.INPUT_DIRECTORY).shape
            self.DIMENSIONS = (shape[1], shape[0])

        self.tresor = np.zeros((self.DIMENSIONS[1], self.DIMENSIONS[0], 3), dtype=np.uint64)
        
        self.stop_time("initialization: {0:.3f}{1}")

        # Curve
        if self.DISPLAY_CURVE or self.APPLY_CURVE:
            self.curve = self.calculate_brightness_curve(self.input_images)
            
            # for item in self.curve:
            #     print(item)

            # sys.exit(0)
            self.stop_time("compute brightness curve: {0:.3f}{1}")

        if self.DISPLAY_CURVE:
            self.display_curve(self.curve)

        if self.ALIGN:
            self.log.debug("translation data: {}".format(self.aligner.TRANSLATION_DATA))
            self.translation_data = json.load(open(self.aligner.TRANSLATION_DATA, "r"))

        # for item in self.curve:
        #     print("{0:20s} | brightness: {1:>3.1f} | luminosity: {2:>3.1f}".format(item["image_name"], item["brightness"], item["luminosity"]))

        # sys.exit(0)

        # print(*self.input_images[0:10], sep="\n")

        for f in self.input_images:
            try:
                self.process(f)
            except Exception as e:
                self.log.error("ERROR in image: {}".format(f))
                raise(e)

        filepath = self.save(fixed_name=self.FIXED_OUTPUT_NAME)

        self.log.info("finished. time total: {}".format(datetime.datetime.now() - self.starttime))
        sys.exit(0)
