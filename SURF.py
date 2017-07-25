import cv2.xfeatures2d
import numpy as np


# Step 1: Image reading
img_object = cv2.imread('test/gitanes.jpg')
img_scene = cv2.imread('test/table.jpg')
# This value determines how many points will be detected (max points, when value = 0)
minHessian = 0

# Step 2: Detect the keypoints and calculate descriptors (feature vectors) using SURF Detector
surf = cv2.xfeatures2d.SURF_create(minHessian)

keypoints_object, descriptors_object = surf.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = surf.detectAndCompute(img_scene, None)


# Step 3: Matching descriptor vectors using FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors_object,descriptors_scene, k=2)

# Need to draw only good matches, so create a mask
# matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test, create mask
good_matches = []
for (m, n) in matches:
    if m.distance < 0.85*n.distance:
        good_matches.append(m)
        # matchesMask[i] = [1, 0]

if len(good_matches) > minHessian:
    src_pts = np.array(np.float32([ keypoints_object[m.queryIdx].pt for m in good_matches ])).reshape(-1,1,2)
    dst_pts = np.array(np.float32([ keypoints_scene[m.trainIdx].pt for m in good_matches ])).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h = img_object.shape[0]
    w = img_object.shape[1]
    pts = np.array(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img_scene, [np.int32(dst)], True, color=(0, 255, 0), thickness=5)

else:
    print("Not enough matches are found - ", len(good_matches), minHessian)
    matchesMask = None
print(len(good_matches), minHessian)



draw_params = dict(matchColor = (0,255,255),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
# H = findHomography(draw_params, scene, CV_RANSAC );


img3 = cv2.drawMatches(img_object,keypoints_object,img_scene,keypoints_scene,good_matches, None, **draw_params)
cv2.namedWindow('Image of object', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image of scene', cv2.WINDOW_NORMAL)
cv2.namedWindow('Result image', cv2.WINDOW_NORMAL)
cv2.imshow('Image of object', img_object)
cv2.imshow('Image of scene', img_scene)
cv2.imshow('Result image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

