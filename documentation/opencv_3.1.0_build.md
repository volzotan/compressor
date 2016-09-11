# build parameters:

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.5/site-packages \
    -D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.5.2/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib \
    -D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/3.5.2/Frameworks/Python.framework/Versions/3.5/include/python3.5m \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_python3=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib/modules ..
