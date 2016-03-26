from PIL import Image
import json, os, sys

"""
    crop loads every image in the directory
    INPUT_DIR, resizes and crops and saves to
    OUTPUT_DIR

    runtime with pypy seems equal.
    
"""
#SIZE_THRESHOLD = 2000000 #150000

INPUT_DIR  = "stack"
OUTPUT_DIR = "cropped_stack"

counter             = 0
skipped             = 0
already_existing    = 0
error               = 0
notajpeg            = 0

filelist = []



# root = INPUT_DIR
# for i in range(0, 10000):
#     f = "{}.jpg".format(i)
#     path = os.path.join(root, f)
#     if os.path.isfile(path):    
#         filelist.append((root, f, os.path.getctime(path)))
        
#filelist.sort(key=lambda tup: tup[1])

INPUT_DIR  = "aligned"
OUTPUT_DIR = "cropped_aligned"

for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if os.path.getsize(os.path.join(root, f)) < 100:
            continue

        if not f.endswith(".jpg"):
            continue

        if counter % 10 == 0:
            filelist.append((root, f))

        counter += 1

counter = 0

for filetuple in filelist:
    root = filetuple[0]
    f = filetuple[1]
    if f.endswith(".jpg"):
        path = os.path.join(root, f)

        # crop_name = os.path.join(OUTPUT_DIR, f)

        # if os.path.isfile(crop_name):
        #     already_existing += 1
        #     continue

        crop_name = os.path.join(OUTPUT_DIR, str(counter) + ".jpg")

        size = os.path.getsize(path)

        # Wie sich rausstellt ist die Komprimierbarkeit des JPGs ein
        # viel besserer Indikator fuer den Bildinhalt als Helligkeit
        # oder Standardabweichung

        # if size < SIZE_THRESHOLD: # 170kb
        #     skipped += 1
        #     continue

        # try:
        #     if img["brightness"] < 40 or img["stddev"] < 30:
        #         continue
        # except Exception as e:
        #     print(e)
        #     print(data[key])
        #     continue

        im = Image.open(path)

        w, h = im.size
        try:
            # 5184 x 3136
            # 1920*2.6 = 4992
            # 1080*2.6 = 2802

            # 5184 - 4992 = 192
            # 3136 - 2802 = 334

            im.crop((170, 260, w-22, h-74)).resize((1920, 1080), Image.BICUBIC).save(crop_name)  # Image.ANTIALIAS

            counter += 1
        except Exception as e:
            print(str(e))
            print(path)
            error += 1

        print("{} cropped {}  skipped {}  already_existing {}  error {}  not-a-jpeg {}".format(f, counter, skipped, already_existing, error, notajpeg))
    else:
        notajpeg += 1