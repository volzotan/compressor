import os
import cv2
import numpy as np

# INPUT_DIR = "/Users/volzotan/Downloads/test/sunday_marie2/captures_2"
# BASE_DIR        = "/Users/volzotan/Documents/trashcam_library/2020-06-27_frauenplan"
BASE_DIR        = "/Users/volzotan/Documents/trashcam_library/2020-07-06_pragerstrasse1"
INPUT_DIR_PAR   = BASE_DIR + "/captures_{}"
# OUTPUT_DIR_PAR  = BASE_DIR + "/test" + "/captures_{}_peaked"
OUTPUT_DIR_PAR  = BASE_DIR + "/captures_{}_peaked"

DIMENSIONS = None

SAVE_EACH_IMAGE = False # True

# ------------------------------------------------------------------------

for i in range(2, 5):

    counter = 0

    INPUT_DIR = INPUT_DIR_PAR.format(i)
    OUTPUT_DIR = OUTPUT_DIR_PAR.format(i)

    print("\n --- {} --- \n".format(INPUT_DIR))

    os.makedirs(OUTPUT_DIR)

    file_list = []

    for f in os.listdir(INPUT_DIR):
        if f == ".DS_Store":
            continue

        if not f.lower().endswith(".jpg"):
            continue

        if os.path.getsize(os.path.join(INPUT_DIR, f)) < 100:
            continue

        file_list.append((INPUT_DIR, f))

    print("loaded {} images".format(len(file_list)))

    file_list = sorted(file_list) #, key=_sort_helper)

    images = file_list

    shape = cv2.imread(os.path.join(*file_list[0])).shape
    DIMENSIONS = (shape[1], shape[0])

    tresor = np.zeros((DIMENSIONS[1], DIMENSIONS[0], 3), dtype=np.uint64)

    for f in images:

        print(f)
        counter += 1

        image = cv2.imread(os.path.join(*f), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 

        # LIGHTEN

        brighter_mask = tresor < image 
        min_brightness_mask = image > 10 #120

        min_brightness_mask = np.any(min_brightness_mask, axis=2, keepdims=True)

        brighter_mask = np.logical_and(brighter_mask, min_brightness_mask)

        tresor[brighter_mask] = image[brighter_mask]

        if SAVE_EACH_IMAGE:
            filepath = os.path.join(OUTPUT_DIR, "output_{}_{:05}.jpg".format(i, counter))
            s = np.asarray(tresor, np.uint16)
            cv2.imwrite(filepath, s)

    filepath = os.path.join(OUTPUT_DIR, "output_{}.jpg".format(i))

    t = tresor.copy()

    overflow_perc = np.amax(t) / (np.iinfo(np.uint64).max / 100.0)
    if overflow_perc > 70:
        print("tresor overflow status: {}%".format(round(overflow_perc, 2)))

    # t = t / (self.counter)

    s = np.asarray(t, np.uint16)
    cv2.imwrite(filepath, s)

print("done.")

