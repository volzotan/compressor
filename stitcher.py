import sys
import time

import cv2
import numpy as np

class Stitcher(object):

    INPUT_DIR           = "images"
    OUTPUT_DIR          = "stitched"

    def __init__(self):
        pass
        

    def init(self):
        self.__init__()


    def _normalize_matrix(self, mat):
        mat /= mat[2, 2]
        return mat


    def mix_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]


        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  \
                        np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                        # print "BLACK"
                        # instead of just putting it with black, 
                        # take average of all nearby values and avg it.
                        warpedImage[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(warpedImage[j,i],[0,0,0])):
                            # print "PIXEL"
                            warpedImage[j,i] = leftImage[j,i]
                        else:
                            if not np.array_equal(leftImage[j,i], [0,0,0]):
                                bl,gl,rl = leftImage[j,i]                               
                                warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage


    def process(self, imageset):
        print(imageset)

        ims = []
        ims_gray = []
        features = []

        stopwatch = time.time()

        for f in imageset:
            im = cv2.imread(f, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ims.append(im)
            ims_gray.append(im_gray)

            sift_obj = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift_obj.detectAndCompute(im_gray, None)
            features.append({"keypoints": keypoints, "descriptors": descriptors})

        print("SIFT: {0:.2}s".format(time.time()-stopwatch))
        stopwatch = time.time()

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        draw_params = {}

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(features[0]["descriptors"], features[1]["descriptors"], k=2)

        good = []
        for item in matches:
            if item[0].distance < 0.7 * item[1].distance:
                good.append(item)

        print("matching: {0:.2}s".format(time.time()-stopwatch))
        stopwatch = time.time()

        print("total matches: {0:4d} | good matches: {1:4d} | ratio: {2:.1f}%".format(len(matches), len(good), len(matches)/len(good)))

        if (len(good) < 4):
            print("too few corresponding points. abort.")
            sys.exit(-1)

        im3 = cv2.drawMatchesKnn(ims[0], features[0]["keypoints"], ims[1], features[0]["keypoints"], good, None, **draw_params)
        cv2.imwrite("correspondences.jpg", im3)

        srcPoints = []
        dstPoints = []

        for item in good:
            srcPoints.append(features[1]["keypoints"][item[0].trainIdx].pt)
            dstPoints.append(features[0]["keypoints"][item[0].queryIdx].pt)

        srcPoints = np.float32(srcPoints)
        dstPoints = np.float32(dstPoints)

        h, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 4)

        self._normalize_matrix(h)

        sys.exit(0)

        dimension_of_warped_image = (5000, 5000)

        warped_image = cv2.warpPerspective(ims[1], h, dimension_of_warped_image)

        cv2.imwrite("warped2.jpg", warped_image)

        warped_image = self.mix_match(ims[0], warped_image)

        cv2.imwrite("warped3.jpg", warped_image)


    def run(self, imagesets):

        for imageset in imagesets:
            self.process(imageset)

            