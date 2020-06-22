# Requirements:

## EXIF related:

brew install pygobject3 --with-python3

    in case pygobject can not be found (AttributeError: module 'gi' has no attribute 'require_version'), 
    install pygobject via pip (requirements need to be installed separately):

    pip install pygobject

brew install tesseract (?)

### ubuntu:
sudo apt-get install python3-gi
sudo apt-get install gir1.2-gexiv2-0.10

sudo pip3 install pillow
sudo pip3 install scipy
sudo pip3 install matplotlib
sudo pip3 install pyyaml