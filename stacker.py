from PIL import Image
import json
import sys, os
import pickle
import datetime

import numpy as np


"""
    Stacker loads every image in the "aligned"
    directory and stacks it. Output to RESULT_DIRECTORY

    Do not run with pypy! (Saving image 30-40s slower)

    numpy for pypy
    pypy -m pip install git+https://bitbucket.org/pypy/numpy.git

"""

starttime = datetime.datetime.now()
timer = datetime.datetime.now()

def stop_time(msg=None):
    global timer
    seconds = (datetime.datetime.now() - timer).total_seconds()
    if msg is not None:
        print(msg.format(seconds))
    timer = datetime.datetime.now()
    return seconds

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

PICKLE_NAME         = "stack.pickle"
DIRECTORY           = "aligned"
RESULT_DIRECTORY    = "stack"
DIMENSIONS          = (5184, 3136) #(1200, 545)

PYPY                = False

change_brightness   = False # should brightness_increase be applied?
BRIGHTNESS_INCREASE = 0.80  # the less the brighter: divider * BRIGHTNESS_INCREASE

SAVE_INTERVAL       = 10
PICKLE_INTERVAL     = 300

#data               = json.load(open("export.json", "rb"))

counter             = 0
processed           = 0

crops               = []
stacked_images      = []

# three dimensional array
# be careful about using types that are big enough to prevent overflows
#tresor = [[[0 for x in range(0,3)] for x in range(DIMENSIONS[1])] for x in range(DIMENSIONS[0])] 
tresor = np.zeros((DIMENSIONS[1], DIMENSIONS[0], 3), dtype=np.uint64)

stop_time("initialization: {}s")

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

    divider = int(round(counter * BRIGHTNESS_INCREASE, 0)) if change_brightness else counter

    endtime = datetime.datetime.now()
    timediff = endtime - timer
    timer = datetime.datetime.now()

    if PYPY:

        stackIm = Image.new("RGB", DIMENSIONS, "black")
        stack = stackIm.load()

        t = tresor.transpose() / divider

        for x in range(0, DIMENSIONS[0]):
            for y in range(0, DIMENSIONS[1]):
                try:
                    stack[x,y] = (t[0][x][y], t[1][x][y], t[2][x][y])
                except Exception as e:
                    print(str(e))
                    raise e

        # stackIm = Image.fromarray(stack)

        #stackIm.save(os.path.join(RESULT_DIRECTORY, str(datetime.datetime.now()) + "__" + str(counter) + "__" + ".jpg"))
        stackIm.save(os.path.join(RESULT_DIRECTORY, str(counter) + ".jpg"))
        stackIm.close()

    else:
        t = tresor / divider

        # array filled with uint64 has to be converted to uint8 to fit JPEG
        stackIm = Image.fromarray(t.astype(np.uint8), "RGB")

        stackIm.save(os.path.join(RESULT_DIRECTORY, str(counter) + ".jpg"))
        stackIm.close()

    timeperimage = timediff/processed if processed != 0 else 0
    processed    = 0 # reset

    print("saved. counter: {} time total: {} saving image: {} time per image: {}".format(counter, timediff, stop_time(), timeperimage.total_seconds()))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# load pickle and init variables
try:
    pick = pickle.load(open(PICKLE_NAME, "rb"))
    stacked_images = pick["stacked_images"]
    tresor = pick["tresor"]
    print("pickle loaded. resume with {} images".format(len(stacked_images)))
except Exception as e:
    print(str(e))

stop_time("pickle loading: {}s")

# get all file names
for root, dirs, files in os.walk(DIRECTORY):
    for f in files:

        if f == ".DS_Store":
            continue

        if os.path.getsize(os.path.join(DIRECTORY, f)) < 100:
            continue

        crops.append(f)

LIMIT = len(crops)

stop_time("searching for files: {}s")
print("number of images: {}".format(LIMIT))

for f in crops:

    counter += 1

    if f in stacked_images:
        continue

    im = Image.open(os.path.join(DIRECTORY, f)) 

    #data = np.array(im, np.int) # 100ms slower per image
    data = np.asarray(im, np.uint64)
    tresor = np.add(tresor, np.uint64(data))

    stacked_images.append(f)
    im.close()

    processed += 1

    if counter >= LIMIT:
        save()
        write_pickle(tresor, stacked_images)
        sys.exit(0)

    if counter % PICKLE_INTERVAL == 0:
        write_pickle(tresor, stacked_images)

    if counter % SAVE_INTERVAL == 0:
        save()

save()
sys.exit(0)