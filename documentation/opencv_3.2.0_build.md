# build parameters:

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.6/site-packages \
    -D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/libpython3.6m.dylib \
    -D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/include/python3.6m \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_text=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib/modules ..


-D BUILD_opencv_text=OFF <-- tesseract module could not be found ( https://github.com/opencv/opencv_contrib/issues/920 )