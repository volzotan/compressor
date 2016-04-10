import cv2
import numpy as np
import os, sys
import datetime
import json
import pyexiv2
import traceback

OUTPUT_STR  = "{0} {1:>5d}  / {2:>5d} | "
OUTPUT_STR += "skipped {3:>4d} | "
OUTPUT_STR += "aligned {4:>4d} | "
OUTPUT_STR += "failed {5:>4d} | "
OUTPUT_STR += "outlier {6:>4d} | "
OUTPUT_STR += "time_align {7:>.1f}"

class Aligner(object):

    # Paths
    REFERENCE_IMAGE                 = "images/DSCF7390.tif"
    EXTENSION                       = ".tif"

    INPUT_DIR                       = "images"
    OUTPUT_DIR                      = "aligned"

    TRANSLATION_DATA                = "translation_data.json"
    JSON_SAVE_INTERVAL              = 100
    SKIP_TRANSLATION                = -1     # do calculate translation data only from every n-th image
    USE_CORRECTED_TRANSLATION_DATA  = False

    # Options
    DOWNSIZE                        = True
    CROP                            = False
    TRANSFER_METADATA               = True
    RESET_MATRIX_EVERY_LOOP         = True
    OUTPUT_IMAGE_QUALITY            = 75    # JPEG
    SIZE_THRESHOLD                  = 100   # bytes

    # ECC Algorithm
    NUMBER_OF_ITERATIONS            = 1000
    TERMINATION_EPS                 = 1e-10
    WARP_MODE                       = cv2.MOTION_TRANSLATION

    def __init__(self):

        self.counter             = 0
        self.skipped             = 0
        self.already_existing    = 0
        self.success             = 0
        self.failed              = 0
        self.outlier             = 0
        
        # Read the reference image (as 8bit for the ECC algorithm)
        self.reference_image = cv2.imread(self.REFERENCE_IMAGE)

        if self.reference_image is None:
            print("reference image not found!")
            sys.exit(-1)

        # Find size
        self.sz = self.reference_image.shape

        if self.DOWNSIZE:
            # proceed with downsized version
            self.reference_image = cv2.resize(self.reference_image, (0,0), fx=0.25, fy=0.25)

        self.reference_image_gray = cv2.cvtColor(self.reference_image,cv2.COLOR_BGR2GRAY)
            
        # Define termination criteria
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.NUMBER_OF_ITERATIONS,  self.TERMINATION_EPS)


    def calculate_translation_values(self, image, warp_matrix):

        source_file = os.path.join(self.INPUT_DIR, image)

        if self.RESET_MATRIX_EVERY_LOOP:
            warp_matrix = self._create_warp_matrix() # reset

        im2 = self._read_image_and_crop(source_file, read_as_8bit=True) 

        # proceed with downsized version
        if self.DOWNSIZE:
            im2_downsized = cv2.resize(im2, (0,0), fx=0.25, fy=0.25)
        else:
            im2_downsized = im2

        im2_gray = cv2.cvtColor(im2_downsized, cv2.COLOR_BGR2GRAY)

        # run ECC
        try:
            (cc, warp_matrix) = cv2.findTransformECC(self.reference_image_gray, im2_gray, warp_matrix, self.WARP_MODE, self.CRITERIA)
        except Exception as e:
            raise e

        if self.DOWNSIZE:
            return (im2, warp_matrix, warp_matrix[0][2] * 4, warp_matrix[1][2] * 4)
        else:
            return (im2, warp_matrix, warp_matrix[0][2], warp_matrix[1][2])


    def transform_and_write(self, image_object, image_name, x, y):

        destination_file    = os.path.join(self.OUTPUT_DIR, image_name)
        warp_matrix         = self._create_warp_matrix()

        warp_matrix[0][2] = x
        warp_matrix[1][2] = y

        if self.WARP_MODE == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (image_object, warp_matrix, (self.sz[1],self.sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(image_object, warp_matrix, (self.sz[1],self.sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        # Write final results
        cv2.imwrite(destination_file, im2_aligned, [int(cv2.IMWRITE_JPEG_QUALITY), self.OUTPUT_IMAGE_QUALITY])


    def step1(self, images):

        self._load_data()

        # Calculate all the translation values and write them into an JSON file

        warp_matrix = self._create_warp_matrix()

        for image in images:
            self.counter += 1

            if image in self.translation_data:
                self.already_existing += 1
                print("{} already calculated".format(image))
                continue

            if os.path.getsize(os.path.join(self.INPUT_DIR, image)) < self.SIZE_THRESHOLD:
                self.skipped += 1
                print("{} empty image".format(image))
                continue

            if self.SKIP_TRANSLATION > 0 and self.success % self.SKIP_TRANSLATION != 0:
                skip = True
            else:
                skip = False

            timer_start = datetime.datetime.now()
            if not skip:
                try:
                    (image_object, new_warp_matrix, translation_x, translation_y) = self.calculate_translation_values(image, warp_matrix)
                except Exception as e:
                    self.failed += 1
                    timediff = datetime.datetime.now() - timer_start
                    print("{} failed [{}s]".format(image, round(timediff.total_seconds(), 2)))
                    tb = traceback.format_exc()
                    print(tb)
                    continue

                # reuse warp matrix for next computation to speed up algorithm
                warp_matrix = new_warp_matrix
            else:
                translation_x = 0
                translation_y = 0

            timediff = datetime.datetime.now() - timer_start
            self.success += 1

            # numpy float32 to python float
            #                               calculated translation in both axes           corrected values
            self.translation_data[image] = ((float(translation_x), float(translation_y)), (0.0, 0.0))

            if not skip:
                print(OUTPUT_STR.format(image, self.counter, len(images), self.skipped, self.success, self.failed, self.outlier, timediff.total_seconds()))

            if self.counter % self.JSON_SAVE_INTERVAL == 0:
                self._save_data()

        self._save_data()


    def step2(self):
        self._load_data()

        images = []

        for item in self.translation_data.keys():
            images.append(item)

        for image in images:
            self.counter += 1

            source_file = os.path.join(self.INPUT_DIR, image)
            destination_file = os.path.join(self.OUTPUT_DIR, image)

            if os.path.isfile(destination_file):
                self.already_existing += 1
                print("{} already transformed".format(image))
                continue

            if image not in self.translation_data:
                self.failed += 1
                print("{} translation data missing".format(image))

            if self.USE_CORRECTED_TRANSLATION_DATA:
                # translation_data[image] = ( (computed_x, computed_y), (corrected_x, corrected_y) ) 
                (x, y) = (self.translation_data[image][1][0], self.translation_data[image][1][1])
            else:
                (x, y) = (self.translation_data[image][0][0], self.translation_data[image][0][1])

            timer_start = datetime.datetime.now()

            im2 = self._read_image_and_crop(source_file)
            self.transform_and_write(im2, image, x, y)

            timediff_align = datetime.datetime.now() - timer_start

            # extract metadata and insert into aligned image
            if self.TRANSFER_METADATA:
                self._transfer_metadata(source_file, destination_file)

            print(OUTPUT_STR.format(image, self.counter, len(images), self.skipped, self.success, self.failed, self.outlier, timediff_align.total_seconds()))


    def _transfer_metadata(self, source, destination):
        metadata_source = pyexiv2.ImageMetadata(source)
        metadata_source.read()
        metadata_destination = pyexiv2.ImageMetadata(destination)
        metadata_destination.read()

        #key_types = [metadata_source.exif_keys()]

        for key in metadata_source.exif_keys:
            try:
                metadata_destination[key] = pyexiv2.ExifTag(key, metadata_source[key].value)
            except Exception as e:
                print(key + "   " + str(e))

        for key in metadata_source.iptc_keys:
            try:
                metadata_destination[key] = pyexiv2.IptcTag(key, metadata_source[key].value)
            except Exception as e:
                print(key + "   " + str(e))

        # for key in metadata_source.xmp_keys:
        #     try:
        #         metadata_destination[key] = pyexiv2.XmpTag(key, metadata_source[key].value)
        #     except Exception as e:
        #         print(key + "   " + str(e))

        metadata_destination.write()


    def _load_data(self):

        # translation_data already existing?

        self.translation_data = {}

        try:
            self.translation_data = json.load(open(self.TRANSLATION_DATA, "rb"))
        except Exception as e:
            print(str(e))


    def _save_data(self):

        json.dump(self.translation_data, open(self.TRANSLATION_DATA, "wb"))
        print("json exported...")


    def _create_warp_matrix(self):
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.WARP_MODE == cv2.MOTION_HOMOGRAPHY:
            return np.eye(3, 3, dtype=np.float32)
        else:
            return np.eye(2, 3, dtype=np.float32)


    def _read_image_and_crop(self, source_file, read_as_8bit=False):

        if not read_as_8bit:
            im = cv2.imread(source_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            im = cv2.imread(source_file) 

        if not self.CROP:
            return im
        else:
            return im[290:3426, 0:5184]



if __name__ == "__main__":

    if (sys.argv) <= 1:
        sys.exit(0)

    if sys.argv[1] == "step1":
        ls = []

        # acquire all filenames
        for root, dirs, files in os.walk(Aligner.INPUT_DIR):
            for f in files:

                if f == ".DS_Store":
                    continue

                if not f.lower().endswith(Aligner.EXTENSION.lower()):
                    continue

                # if f in problem_list:
                #     continue

                ls.append(f)

        Aligner().step1(ls)
    elif sys.argv[1] == "step2":
        Aligner().step2()
    else:
        print("illegal argument. exit.")
        