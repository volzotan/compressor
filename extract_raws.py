import os
import argparse

from pydng.core import RPICAM2DNG

MIN_RAW_SIZE_IN_BYTES = 1024*1024*10 

filenames = []

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory containing JPEGs with RAW data")
args = parser.parse_args()

for root, dirs, files in os.walk(args.directory):
    for file in files:
        if file.endswith('.jpg'):
            filenames.append([os.path.join(root, *dirs), file])

filenames = sorted(filenames)

d = RPICAM2DNG()
for f in filenames:
    p = os.path.join(*f)

    if os.path.getsize(p) < MIN_RAW_SIZE_IN_BYTES:
        print("skip:    {}".format(p))
        continue

    d.convert(p, compress=False)

    print("convert: {}".format(p))
