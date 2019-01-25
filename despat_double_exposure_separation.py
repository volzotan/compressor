import os
import shutil

INPUT_FOLDER    = "/Volumes/ctdrive/export_bangkok120_jpeg"
OUTPUT_FOLDER   = os.path.join(INPUT_FOLDER, "2nd")

EXTENSION = ".jpg"

MAX_TIME_DIFF = 3000

images = []
double_exposure_images = []

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

prev = None
for img in images:
    timestamp = os.path.splitext(img[1])[0]
    timestamp = timestamp[:timestamp.rfind("_")]
    timestamp = int(timestamp)

    if prev is not None:
        if (timestamp - prev[0]) < MAX_TIME_DIFF:
            double_exposure_images.append((prev[1], img))

    prev = (timestamp, img)


for img in double_exposure_images:
    src = os.path.join(img[1][0], img[1][1])
    dst = os.path.join(OUTPUT_FOLDER, img[0][1])
    shutil.move(src, dst)

print(double_exposure_images)
