import os
import shutil

INPUT_FOLDER    = "/Users/volzotan/Downloads/despat/despat"
OUTPUT_FOLDER   = os.path.join(INPUT_FOLDER, "2nd")

EXTENSION = ".jpg"

MAX_TIME_DIFF = 3000

images = []
double_exposure_images = []

def _sort_helper(d):

    # still_123.jpg

    value = d[1]

    if value.startswith("still_"):
        pos = value.index(".")
        number = value[6:pos]
        return int(number)        
    elif value.startswith("DSCF"):
        pos = value.index(".")
        number = value[4:pos]
        return int(number)
    elif value.startswith("DSC"):
        pos = value.index(".")
        number = value[3:pos]
        return int(number)
    elif value.startswith("DJI_"):
        pos = value.index(".")
        number = value[4:pos]
        return int(number)
    else:
        try:
            filename = os.path.splitext(value)[0]
            return int(filename)
        except ValueError as e:
            return 0


# recursive with sub directories
# for root, dirs, files in os.walk(INPUT_FOLDER):
#     for f in files:
#         if f.lower().endswith(EXTENSION):
#             images.append((root, f))

# non-recursive
for f in os.listdir(INPUT_FOLDER):
    if f.lower().endswith(EXTENSION):
        images.append((INPUT_FOLDER, f))

# print(images)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print("created output dir: {}".format(OUTPUT_FOLDER))

images = sorted(images, key=_sort_helper)

prev = None
for img in images:
    timestamp = os.path.splitext(img[1])[0]
    timestamp = timestamp[:timestamp.rfind("_")]
    timestamp = int(timestamp)

    if prev is not None:
        diff = timestamp - prev[0]
        if diff < MAX_TIME_DIFF:
            double_exposure_images.append((prev[1], img))
            # print(diff)

    prev = (timestamp, img)


# save the 2nd file with the filename of the prev file in the output folder

for img in double_exposure_images:
    src = os.path.join(img[1][0], img[1][1])
    dst = os.path.join(OUTPUT_FOLDER, img[0][1])
    shutil.move(src, dst)

# print(double_exposure_images)
